from pydantic import BaseModel, Field
from sandbox.utils import generate_with_single_input, generate_params_dict


def decide_if_technical_or_creative(query):
    """
    Determines whether a given query is creative or technical in nature.

    Args:
        query (str): The query string to be evaluated.

    Returns:
        str: A label indicating the query type, either 'creative' or 'technical'.

    This function constructs a prompt to classify a query based on its content.
    Creative queries typically involve requests to generate original content, whereas
    technical queries relate to documentation or technical information, such as procedures.
    By leveraging an LLM, it identifies the query type and returns an appropriate label.
    """

    PROMPT = f"""Decide if the following query is a creative query or a technical query.
    Creative queries ask you to create content, while technical queries are related to documentation or technical requests, like information about procedures.
    Answer only 'creative' or 'technical'.
    Query: {query}
    """
    result = generate_with_single_input(PROMPT)
    label = result['content']
    return label


def answer_query(query):
    """
    Processes a query and generates an appropriate response by categorizing the query
    as either 'technical' or 'creative', and modifies behavior based on this categorization.

    Args:
        query (str): The query string to be answered.

    Returns:
        str: A generated response from the LLM tailored to the nature of the query.

    This function first determines the nature of the query using the `decide_if_technical_or_creative` function.
    If the query is classified as 'technical', it sets parameters suitable for precise and low-variability responses.
    If the query is 'creative', it applies parameters allowing for more variability and creativity.
    If the classification is inconclusive, it uses neutral parameters.
    It then generates a response using these parameters and returns the content.
    """

    # Determine whether the query is 'technical' or 'creative'
    label = decide_if_technical_or_creative(query).lower()

    # Set parameters for technical queries (precise, low randomness)
    if label == 'technical':
        kwargs = generate_params_dict(query, temperature=0, top_p=0.1)

    # Set parameters for creative queries (variable, high randomness)
    elif label == 'creative':
        kwargs = generate_params_dict(query, temperature=1.1, top_p=0.4)

    # Use default parameters if the query type is inconclusive
    else:
        kwargs = generate_params_dict(query, temperature=0.5, top_p=0.5)

    # Generate a response based on the query type and parameters
    response = generate_with_single_input(**kwargs)

    # Extract and return the content from the response
    result = response['content']
    return result


def generate_system_call(command):
    PROMPT = f"""
You are an assistant program that converts natural language commands into structured JSON for controlling smart home devices. The JSON should conform to a specific format describing the device, action, and parameters. Here's how you can do it:

**Available Devices and Actions:**

1. **Light**
   - Actions: "turn on", "turn off"
   - Parameters: color, intensity (percentage)

2. **Automatic Lock**
   - Actions: "lock", "unlock"
   - Parameters: None

3. **Sound System (Speaker)**
   - Actions: "play", "pause", "stop", "set volume"
   - Parameters: volume (integer), track (string), playlist_style (string)

4. **TV**
   - Actions: "turn on", "turn off", "change channel", "adjust volume"
   - Parameters: channel (string), volume (integer)

5. **Air Conditioner**
   - Actions: "turn on", "turn off", "set temperature", "adjust fan speed"
   - Parameters: temperature (integer), fan_speed (low/medium/high)

**Rooms and Devices:**
- **Office**
  - Lights: "office_light_1" (ID: 123), "office_light_2" (ID: 321)
  - Automatic Lock: "office_door_lock" (ID: 111)

- **Living Room**
  - Light: "living_room_light" (ID: 222)
  - Speaker: "living_room_speaker" (ID: 223)
  - Air Conditioner: "living_room_airconditioner" (ID: 556)

- **Kitchen**
  - Light: "kitchen_light" (ID: 333)

- **Bedroom**
  - Light: "bedroom_light" (ID: 444)
  - TV: "bedroom_tv" (ID: 445)

- **Bathroom**
  - Light: "bathroom_light" (ID: 555)

**Task:**
Convert the following natural language command into the structured JSON format based on the available devices:

**Input Examples:**

1. "Turn on the office light with ID 123 with blue color and 50% intensity."
   - JSON:
     [
     {{
       "room": "office",
       "object_id": "123",
       "object_name": "office_light_1",
       "action": "turn on",
       "parameters": {{"color": "blue", "intensity": "50%"}}
     }}
     ]

2. "Lock the office door."
   - JSON:
   [
     {{
       "room": "office",
       "object_id": "111",
       "object_name": "office_door_lock",
       "action": "lock",
       "parameters": {{}}
     }}
    ]

2. "Make my living room a cheerful place"
   - JSON:
   [
     {{
       "room": "living_room",
       "object_id": "222",
       "object_name": "living_room_light",
       "action": "turn on",
       "parameters": {{'intensity': '80%', 'color':'yellow'}}
     }},
     {{
       "room": "living_room",
       "object_id": "223",
       "object_name": "living_room_speaker",
       "action": "turn on",
       "parameters": {{'volume': '100', 'playlist_style':'party'}}
     }},

   ]

**Note:**
- Ensure that each JSON object correctly maps the natural command to the appropriate device and action using the listed device ID.
- Use the object ID to differentiate between devices when the room contains multiple similar items.
- You can add more than one parameter in the parameters dictionary.

Using this information, translate the following command into JSON: "{command}". Output a list with all the necessary JSONs. 
Always output a list even if there is only one command to be applied, do not output anything else but the desired structure.
"""
    kwargs = generate_params_dict(PROMPT, temperature=0.4, top_p=0.1)
    result = generate_with_single_input(**kwargs)
    return result['content']


class VoiceNote(BaseModel):
    title: str = Field(description="A title for the voice note")
    summary: str = Field(description="A short one sentence summary of the voice note.")
    actionItems: list[str] = Field(
        description="A list of action items from the voice note"
    )
