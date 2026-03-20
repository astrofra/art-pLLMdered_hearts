import ollama
import pexpect
import time
import re
import json
import signal
import os
import select

DEFAULT_OLLAMA_MODEL = "ministral-3:14b"
GAME_CONTEXT = (
    "Plundered Hearts is an Infocom text adventure set in the Caribbean. "
    "You play a kidnapped young woman aboard a pirate ship and must suggest the next text command."
)
ZMACHINE_PARSER_GUIDANCE = (
    "The game uses a classic Infocom-style parser. "
    "Prefer very short commands: most valid inputs are one or two words. "
    "Use exactly one action per turn. "
    "Directions are usually single letters like N, S, E, W, and vertical moves like UP or DOWN. "
    "Wait is often Z. "
    "Default command shape is VERB or VERB NOUN, such as EXAMINE DOOR or TAKE BOTTLE. "
    "When needed, use simple parser patterns like VERB NOUN PREPOSITION NOUN, for example LOOK THROUGH WINDOW, "
    "PUT RAG IN WATER, THROW PORK AT CROCODILE, or UNLOCK DOOR WITH KEY. "
    "Use nouns that appear explicitly in the latest game text. "
    "Prefer concrete adventure verbs like EXAMINE, LOOK, OPEN, TAKE, READ, ENTER, CLIMB, PUSH, PULL, GIVE, THROW, "
    "WEAR, REMOVE, UNLOCK, DRINK, ASK, or TALK. "
    "Do not write full sentences, explanations, punctuation-heavy text, or multiple actions joined together. "
    "Avoid inventing synonyms or abstract verbs if the object names are not present in the scene. "
    "If the game asks a yes or no question addressed to a character, the parser may accept a compact reply like NAME, YES or NAME, NO."
)

class OllamaTimeoutError(Exception):
    pass

def extract_and_parse_json(text):
    """
    Extracts the first JSON object found inside triple backticks or code-like blocks from the input text,
    and returns it as a Python dictionary.
    """

    # Try to find JSON inside ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        # If no fenced block, try raw { ... } block (useful for degraded format)
        match = re.search(r"(\{[\s\S]*?\})", text)
    if not match:
        raise RuntimeError("No JSON object found in Ollama response.")

    json_str = match.group(1)
    return json.loads(json_str)

ansi_escape = re.compile(r'''
    \x1b    # ESC
    \[      # CSI
    [0-9;]* # Parameters
    [A-Za-z] # Command letter
''', re.VERBOSE)

cursor_directives = re.compile(r'\[\d{1,3}d')
charset_switch = re.compile(r'\x1b\([A-B]')

def clean_output(text):
    # Cleanup ANSI (line feed, cursor, etc.)
    text = ansi_escape.sub('', text)
    # Cleanup "[24d" and alike
    text = charset_switch.sub('', text)
    text = cursor_directives.sub('', text)
    # Cleanup score (attempt to...)
    lines = text.strip().splitlines()
    lines = [line for line in lines if not line.strip().isdigit()]
    return '\n'.join(lines).strip()


def pick_ollama_model():
    preferred_models = [
        DEFAULT_OLLAMA_MODEL,
        "ministral-3:14b",
        "minimax-m2.7:cloud",
        "qwen3-vl:8b",
    ]

    available_models = [model.model for model in ollama.list().models]

    for model in preferred_models:
        if model in available_models:
            return model

    for model in available_models:
        if "embed" not in model:
            return model

    return DEFAULT_OLLAMA_MODEL


def read_game_output(child, timeout=2):
    output = ""
    idle_cycles = 0

    while True:
        ready, _, _ = select.select([child.child_fd], [], [], timeout)
        if not ready:
            idle_cycles = idle_cycles + 1
            if idle_cycles >= 2:
                break
            continue

        chunk = os.read(child.child_fd, 4096)
        if not chunk:
            break
        output = output + chunk.decode("utf-8", errors="ignore")
        idle_cycles = 0

        while "[MORE]" in output or "***MORE***" in output:
            output = output.replace("[MORE]", "")
            output = output.replace("***MORE***", "")
            child.send(" ")

    return clean_output(output)

def _handle_ollama_timeout(signum, frame):
    raise OllamaTimeoutError()


def ask_ollama_for_command(model, prompt, timeout_seconds=20):
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_ollama_timeout)
    signal.alarm(timeout_seconds)

    response = ollama.chat(
        model=model,
        format='json',
        options={'temperature': 0.2},
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous_handler)
    # print(response.message.content)
    return extract_and_parse_json(response.message.content)


# official Amiga solution
plundered_hearts_commands = [
    "stand up", "inventory", "examine smelling salts", "read tag", "examine banknote",
    "examine coffer", "examine door", "open door", "z", "scream", "e", "examine falcon",
    "read missive", "falcon, yes", "examine davis", "examine ring", "examine falcon", "z", "z",
    "stand up", "look around", "look through window", "open curtain", "examine cupboard",
    "examine table", "examine bed", "z", "examine brooch", "open coffer", "take invitation",
    "read invitation", "n", "down", "n", "examine gate", "n", "take bottle", "take mirror",
    "examine bottle", "read label", "s", "s", "up", "open door", "enter", "take clothes",
    "remove dress", "wear breeches", "wear shirt", "z", "out", "s", "take coffer", "z",
    "throw coffer through window", "sit on ledge", "put all in reticule", "take ladder",
    "up", "up", "up", "up", "n", "n", "n", "examine winch", "read lever", "pull lever up",
    "s", "examine barrels", "tear dress", "put rag in water", "open hatch", "down",
    "throw rag over gate", "up", "s", "examine casks", "n", "n", "enter shack", "take dagger",
    "e", "s", "s", "look in cask", "enter cask", "take pork", "put all in reticule except dagger",
    "cut rope", "examine pork", "z", "z", "z", "leave cask", "examine skiff", "w", "n", "e",
    "pull slat", "z", "z", "falcon, yes", "z", "e", "n", "open window", "w", "examine portrait",
    "examine bookshelves", "examine globe", "take hat", "s", "w", "e", "examine lucy",
    "take garter", "w", "s", "ne", "climb vine", "remove clothes", "wear ball gown",
    "take invitation", "put garter in reticule", "n", "n", "take pistols", "s", "e", "e",
    "open door", "s", "n", "w", "down", "give invitation", "s", "dance", "dance", "dance",
    "dance", "w", "examine orchestra", "e", "dance", "examine lafond", "examine ring",
    "open door", "s", "n", "n", "e", "s", "e", "n", "n", "take treatise", "take hat",
    "press sinistra on globe", "n", "down", "e", "e", "take key and horn", "w", "s",
    "open door", "e", "w", "n", "w", "s", "squeeze bottle on pork", "throw pork at crocodile",
    "z", "z", "s", "w", "unlock door", "open door", "n", "give garter to papa", "z", "s",
    "e", "n", "n", "up", "s", "s", "n", "n", "s", "s", "s", "n", "n", "s", "up", "e",
    "knock on door", "open door", "n", "drink wine", "pour wine into green goblet",
    "squeeze bottle into green goblet", "pour wine into blue goblet", "z", "lafond, no",
    "drink wine", "take spice", "throw spice at lafond", "wave mirror at window", "s",
    "w", "down", "s", "z", "cookie, yes", "n", "e", "n", "take treatise", "take hat",
    "press sinistra", "n", "down", "s", "s", "take rapier", "kill crulley", "g",
    "close trapdoor", "pick lock", "wave smelling salts", "n", "n", "z", "up", "s",
    "s", "w", "up", "e", "s", "untie rope", "climb down rope", "take horn", "s", "s", "s",
    "look at nicholas", "push nicholas", "nicholas, yes", "take pistol", "load pistol",
    "fire pistol at crulley"
]

plundered_hearts_solution = "Start in the ship's cabin.\
STAND  UP  -  (you  get out of bed) - INVENTORY - (you find that you have a\
reticule  containing  smelling  salts  and  a bank note) - EXAMINE SMELLING\
SALTS - READ TAG - EXAMINE BANKNOTE - (the ship lurches and a coffer slides\
out  from  under the bed) - EXAMINE COFFER - EXAMINE DOOR - OPEN DOOR - Z -\
(Crulley  bursts  in!)  -  SCREAM  -  E - (Crulley stops you but the Falcon\
enters  and  clouts  him!)  -  EXAMINE  FALCON - (he eventually hands you a\
missive)  -  READ MISSIVE - (he offers you protection) - FALCON, YES - (you\
are now on the Deck).\
\
EXAMINE  DAVIS  - (you are taken to the Captain's Quarters.........two days\
pass by) - EXAMINE RING - EXAMINE FALCON - Z - Z - STAND UP - LOOK AROUND -\
LOOK  THROUGH  WINDOW  -  OPEN CURTAIN - EXAMINE CUPBOARD - EXAMINE TABLE -\
EXAMINE  BED  - Z - (Captain Jamison gives you a brooch) - EXAMINE BROOCH -\
OPEN  COFFER - TAKE INVITATION - READ INVITATION - N - (you squeeze through\
to the Landing) - DOWN - N - EXAMINE GATE - N - TAKE BOTTLE - TAKE MIRROR -\
EXAMINE BOTTLE - READ LABEL - S - S - UP - OPEN DOOR - ENTER - TAKE CLOTHES\
- REMOVE DRESS - WEAR BREECHES - WEAR SHIRT - Z - OUT.\
\
S  - TAKE COFFER - Z - THROW COFFER THROUGH WINDOW - SIT ON LEDGE - PUT ALL\
IN  RETICULE - TAKE LADDER - (you now hang on the ladder!) - UP - UP - UP -\
UP - (you land on the Poop Deck) - N - N - N - EXAMINE WINCH - READ LEVER -\
PULL LEVER UP - (you lower the anchor) - S - EXAMINE BARRELS - TEAR DRESS -\
PUT RAG IN WATER - OPEN HATCH - DOWN - THROW RAG OVER GATE - (you douse the\
burning  fuse) - UP - S - EXAMINE CASKS - N - N - ENTER SHACK - TAKE DAGGER\
-  E  - S - S - LOOK IN CASK - ENTER CASK - TAKE PORK - PUT ALL IN RETICULE\
EXCEPT  DAGGER  - CUT ROPE - (you release the cask into the sea!) - EXAMINE\
PORK - Z - Z - Z - (you eventually drift to some shallows near an Island).\
\
LEAVE CASK - EXAMINE SKIFF - W - N - E - PULL SLAT - (Captain Jamison finds\
you!)  - Z - Z - (he asks to kiss you) - FALCON, YES - (now, now.......it's\
only  a  game  fellas!!)  - Z - (he now leaves) - E - N - OPEN WINDOW - W -\
EXAMINE  PORTRAIT  -  EXAMINE BOOKSHELVES - EXAMINE GLOBE - TAKE HAT - (you\
feel a vibration from the floor!.......ooer!!) - S - (the Butler now throws\
you out!!) - W - E - EXAMINE LUCY - TAKE GARTER - W - S - NE - CLIMB VINE -\
(to a Bedroom).\
\
REMOVE  CLOTHES - WEAR BALL GOWN - TAKE INVITATION - PUT GARTER IN RETICULE\
-  N  -  N  -  TAKE  PISTOLS  -  (out  of reach!) - S - E - E - OPEN DOOR -\
(locked!)  - S - N - W - DOWN - GIVE INVITATION - (to Butler) - S - DANCE -\
DANCE  - DANCE - DANCE - W - EXAMINE ORCHESTRA - E - DANCE - EXAMINE LAFOND\
-  EXAMINE RING - OPEN DOOR - S - N - N - E - (the Butler stops you!) - S -\
E  -  N - (under the table) - N - TAKE TREATISE - TAKE HAT - PRESS SINISTRA\
ON GLOBE - (the portrait opens!) - N - DOWN - E - E - TAKE KEY AND HORN.\
\
W  -  S  - OPEN DOOR - E - W - N - W - S - (you see a crocodile!) - SQUEEZE\
BOTTLE  ON  PORK - THROW PORK AT CROCODILE - Z - Z - (it sleeps!) - S - W -\
UNLOCK  DOOR - (with large key) - OPEN DOOR - N - GIVE GARTER TO PAPA - Z -\
S - E - N - N - UP - S - S - (Captain Jamison is being arrested!) - N - N -\
S - S - S - N - N - S - UP - E - KNOCK ON DOOR - OPEN DOOR - N - DRINK WINE\
-  POUR  WINE  INTO  GREEN GOBLET - SQUEEZE BOTTLE INTO GREEN GOBLET - POUR\
WINE  INTO BLUE GOBLET - Z - (Lafond asks you if the green goblet is his) -\
LAFOND, NO - (he takes the green goblet!) - DRINK WINE - TAKE SPICE - THROW\
SPICE AT LAFOND - WAVE MIRROR AT WINDOW - (to signal the Helena Louise).\
\
S  -  (the  Butler  is  sound  asleep!) - W - DOWN - S - Z - (Jamison's men\
enter!)  -  COOKIE,  YES  -  N  -  E - N - TAKE TREATISE - TAKE HAT - PRESS\
SINISTRA  -  N  -  DOWN - S - (Cookie deals with the crocodile!) - S - TAKE\
RAPIER  -  KILL CRULLEY - G - CLOSE TRAPDOOR - PICK LOCK - (with the brooch\
clasp)  -  WAVE  SMELLING SALTS - N - N - Z - UP - S - S - W - UP - E - S -\
UNTIE  ROPE - CLIMB DOWN ROPE - (you crash into Lafond!!) - (you eventually\
end  up  back in the Ballroom) - TAKE HORN - S - S - S - LOOK AT NICHOLAS -\
PUSH  NICHOLAS - NICHOLAS, YES - TAKE PISTOL - LOAD PISTOL - FIRE PISTOL AT\
CRULLEY............to complete the adventure!!!!!!!!"

# wikipedia article about Plundered Hearts
plundered_hearts_wiki = """Plundered Hearts is an interactive fiction video game created by Amy Briggs and published by Infocom in 1987. Infocom's only game in the romance genre, it was released simultaneously for the Apple II, Commodore 64, Atari 8-bit computers, Atari ST, Amiga, Mac, and MS-DOS. It is Infocom's 28th game.
Plundered Hearts casts the player in a well-defined role. The lead character is a young woman in the late 17th century who has received a letter. Jean Lafond, the governor of the small West Indies island of St. Sinistra, says that the player's father has contracted a wasting tropical disease. Lafond suggests that his recovery would be greatly helped by the loving presence of his daughter, and sends his ship (the Lafond Deux) to transport her.
As the game begins, the ship is attacked by pirates and the player's character is kidnapped. Eventually, the player's character finds that two men are striving for her affections: dashing pirate Nicholas Jamison, and the conniving Jean Lafond. As the intrigue plays out, the lady does not sit idly by and watch the men duel over her; she must help Jamison overcome the evil plans of Lafond so that they have a chance to live happily ever after.
As early as 1984, Infocom employees joked about the possibility of a romance text adventure. By 1987, the year of Plundered Hearts's release, Infocom no longer rated its games on difficulty level.
Although this was not the only Infocom game designed in an effort to attract female players (one example being Moonmist), it is the only game where the lead character is always female.
The Plundered Hearts package included an elegant velvet reticule (pouch) containing the following items:
- A 50 guinea banknote from St. Sinistra
- A letter from Jean Lafond reporting the illness of the player character's father
Game reviewers complimented Plundered Hearts for its gripping prose, challenging predicaments, and scenes of derring-do. Other publications said it was a good introduction to interactive fiction, with writing suitable for both men and women. Some noted that the genre might have alienated Infocom's typical audience, but praised its bold direction nonetheless.
"""

# run dfrotz to get a plain terminal stream that is easier to parse than curses output
child = pexpect.spawn("dfrotz roms/PLUNDERE.z3", encoding='utf-8', timeout=5)
ollama_model = pick_ollama_model()
step = 0

# Start the story and consume intro pages until the command prompt appears.
child.sendline("")
time.sleep(0.5)
print(read_game_output(child))
print("\n")

prev_output = ""
prev_cmd = None

# automated walkthrough
while True : # for step, cmd in enumerate(plundered_hearts_commands):
    step = step + 1
    # print(f"\n→ Étape {step+1}: {cmd}")
    # print(f"{cmd}")
    prompt = "You are playing Plundered Hearts, a text interactive fiction by Amy Briggs. "
    prompt = prompt + GAME_CONTEXT
    prompt = prompt + " "
    prompt = prompt + ZMACHINE_PARSER_GUIDANCE
    prompt = prompt + " Latest game output: "
    prompt = prompt + prev_output
    if prev_cmd is not None:
        prompt = prompt + " Previous command: " + prev_cmd + "."
    prompt = prompt + " Reply with JSON only using keys 'comment' and 'prompt'."
    prompt = prompt + " The 'prompt' value must be a single parser command, not a sentence."

    json_command = ask_ollama_for_command(ollama_model, prompt)

    if not isinstance(json_command, dict):
        raise RuntimeError(f"Unexpected Ollama payload type: {type(json_command).__name__}")
    if "comment" not in json_command or "prompt" not in json_command:
        raise RuntimeError("Ollama JSON must contain both 'comment' and 'prompt'.")

    # print("\n")
    print("AI thinks : '" + json_command["comment"] + "'\n")
    command = json_command["prompt"]
    command = command.replace(">", "").strip().upper()
    child.sendline(command)
    prev_cmd = command
    prev_output = read_game_output(child)
    print(prev_output)

    # time.sleep(0.3)  # artificially wait to allow reading