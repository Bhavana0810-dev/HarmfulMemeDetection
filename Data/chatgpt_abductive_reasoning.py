import json
import time
import openai
import pickle

# Set your OpenAI API key here
openai.api_key = "XXXXX"

# Initialize variables
data = dict()
id = 0

# Load the meme training data from the provided JSON file
with open("./FHM/mem_train.json", "r", encoding='utf8') as fin:
    data_list = json.load(fin)
    # Populate the 'data' dictionary with meme data, using the meme id as key
    for data_item in data_list:
        data[id] = data_item
        id += 1

# Load the captions from the pickle file
with open('./FHM/captions.pkl', 'rb') as f:
    caption_dict = pickle.load(f)

# Prepare the list of meme IDs
cids = list(data.keys())
pred = {}

# Define the system prompt for OpenAI GPT-3.5
system_prompt = (
    "You have been specially designed to perform abductive reasoning for the harmful meme detection task. "
    "Your primary function is that, according to a Harmfulness label about an Image with a text embedded, "
    "please provide me a streamlined rationale, without explicitly indicating the label, why it is classified as the given Harmfulness label. "
    "The image and the textual content in the meme are often uncorrelated, but its overall semantics is presented holistically. "
    "Thus it is important to note that you are prohibited from relying on your own imagination, as your goal is to provide the most accurate and reliable rationale possible "
    "so that people can infer the harmfulness according to your reasoning about the background context and relationship between the given text and image caption."
)

# Start processing memes one by one
count = 0
while count < len(cids):

    cid = cids[count]
    try:
        # Retrieve meme text and caption
        text = data[cid]["clean_sent"].replace('\n', ' ').strip('\n')
        caption = caption_dict[data[cid]["img"].strip('.png')].strip('\n')
        
        # Set the harmfulness label (harmful or harmless)
        label = 'harmful' if data[cid]["label"] == 1 else 'harmless'

        # Prepare the user prompt to send to GPT-3.5
        user_prompt = f"Given a Text: '{text}', which is embedded in an Image: '{caption}'; and a harmfulness label '{label}', please give me a streamlined rationale associated with the meme, without explicitly indicating the label, why it is reasoned as {label}."

        # Send the request to OpenAI API to get the rationale
        reply = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=256,
        )

        # Extract the rationale from the API response
        ans = reply["choices"][0]["message"]["content"]
        print(user_prompt)
        print(ans)
        print(count)

        # Store the rationale in the 'pred' dictionary with the image filename as key
        pred[data[cid]["img"].strip('.png')] = ans.lower()

        # Save the updated predictions to a pickle file
        with open("clipcap_FHM_rationale_.pkl", "wb") as fout:
            pickle.dump(pred, fout)
        
        # Increment the counter to process the next meme
        count += 1

    except Exception as e:
        # If an error occurs, print the message and wait for a while to avoid hitting rate limits
        print(f"Error processing meme {cid}: {e}. Let's have a sleep.")
        time.sleep(61)  # Sleep for 61 seconds to avoid hitting the rate limit
