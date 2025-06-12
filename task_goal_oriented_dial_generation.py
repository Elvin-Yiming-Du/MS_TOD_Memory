import os
import json
import sys
from api_llm_utils import gpt4o_generate

def llm_response(prompt,
                 system_prompt="You are a helpful assistant, please help me generate the response as required by user.", max_length=3000, temperature=0.9):
    response = gpt4o_generate(prompt, system_prompt=system_prompt, max_length=max_length, temperature=temperature)
    return response

def parse_response_results(llm_response, sample_key_words):
    sample_intent = sample_key_words["intent"]
    sample_domain = sample_key_words["domain"]
    sample_reference_dialogue_id = sample_key_words["reference_dialogue_id"]
    llm_response = llm_response.replace("```json\n", "").replace("\n```", "").replace("\n", "").strip()
    conversations = json.loads(llm_response)
    parsed_conversations = []
    output_sessions = []
    for session_id, session in enumerate(conversations["sessions"], 1):
        temp_session = {}
        temp_session["session_id"] = session_id
        temp_session["domain"] = sample_domain
        temp_session["reference_dialogue_id"] = sample_reference_dialogue_id
        temp_session["exist_confirmation"] = False
        temp_session["content"] = []
        temp_session["intent"] = sample_intent
        temp_session["task_goal"] = []

        print(f"\nSession {session_id}:")
        utterance_id = 0
        for turn in session:
            speaker = turn["speaker"]
            text = turn["text"]
            print(f"{speaker.capitalize()}: {text}")
            
            # Check for confirmation-type utterances
            if "is_confirmation" in turn and turn["is_confirmation"] == True:
                temp_session["exist_confirmation"] = True
                temp_session["task_goal"].append({"utterance_id": utterance_id, "slot_values": sample_key_words["task_goal"]})
                temp_session["content"].append({"speaker": turn["speaker"], "utterance": turn["text"], "is_confirmation": True})
            else:
                temp_session["content"].append({"speaker": turn["speaker"], "utterance": turn["text"], "is_confirmation": False})
            utterance_id += 1
        output_sessions.append(temp_session)
    return output_sessions

def read_generated_data(file_path):
    # Load the JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract dialogue IDs and count occurrences of new_session_3 and new_session_4
    dialogue_ids = []
    # Loop through each dialogue entry
    for dialogue in data.values():
        # Extract dialogue_id
        for session in dialogue["sessions"]:
            dialogue_ids.append(session["reference_dialogue_id"])

    # # Load the JSON data
    # with open("raw_gpt4_0923_3000.json", 'r', encoding='utf-8') as f:
    #     data2 = json.load(f)


    # # Loop through each dialogue entry
    # for dialogue in data2.values():
    #     # Extract dialogue_id
    #     dialogue_ids.append(dialogue.get('dialogue_id'))

    return dialogue_ids


def load_sgd_intents(folder_path):
    # 遍历文件夹中的所有文件
    all_intents = []
    sgd_intents_link = []
    domain_intent_examples = {}
    exception_number = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            for dialogue in data:
                try:
                    if "services" in dialogue.keys():
                        exp_domain = "services"
                    elif "service_name" in dialogue.keys():
                        exp_domain = "service_name"
                    else:
                        exp_domain = "services_name"

                    exp_domains = search_domain(dialogue[exp_domain])
                    if len(exp_domains) >= 1:
                        for exp_domain in exp_domains:
                            if exp_domain not in domain_intent_examples.keys():
                                domain_intent_examples[exp_domain] = {}
                        turn_intents = []
                        for turn in dialogue["turns"]:
                            if "state" in turn["frames"][0]:
                                service_name = turn["frames"][0]["service"]
                                service_domain = search_domain([service_name])[0]
                                intent = turn["frames"][0]["state"]["active_intent"]
                                if intent == "NONE":
                                    continue
                                turn_intents.append(intent)
                                all_intents.append(intent)
                                if intent not in domain_intent_examples[service_domain]:
                                    domain_intent_examples[service_domain][intent] = {}

                                if dialogue["dialogue_id"] not in domain_intent_examples[service_domain][intent]:
                                    domain_intent_examples[service_domain][intent][dialogue["dialogue_id"]] = dialogue[
                                        "turns"]
                        sgd_intents_link.append(list(set(turn_intents)))
                except Exception as e:
                    print(e)
    all_intents = list(set(all_intents))

    return domain_intent_examples, all_intents, sgd_intents_link

# def llm_response(prompt,
#                  system_prompt="You are a helpful assistant, please help me generate the response as required by user.", max_length=3000, temperature=0.9):
#     response = get4mini_generate(prompt, system_prompt=system_prompt, max_length=max_length, temperature=temperature)
#     return response

def split_dialogue_content_by_intent(dialogue_content):
    all_dialogues = []
    temp_dialogues = []
    current_dialogue_intent = ""
    for dialogue in dialogue_content:
        if "state" in dialogue["frames"][0].keys():
            active_intent = dialogue["frames"][0]["state"]["active_intent"]
            if current_dialogue_intent != active_intent and current_dialogue_intent!="" and active_intent!= "NONE":
                all_dialogues.append({current_dialogue_intent: temp_dialogues})
                current_dialogue_intent = active_intent
                temp_dialogues = []
            elif current_dialogue_intent != active_intent and current_dialogue_intent=="" and active_intent!= "NONE":
                current_dialogue_intent = active_intent
                temp_dialogues.append(dialogue)
            else:
                temp_dialogues.append(dialogue)
        else:
            temp_dialogues.append(dialogue)
    if len(temp_dialogues) != 0:
        all_dialogues.append({current_dialogue_intent: temp_dialogues})
    return all_dialogues

def extract_slot_values_from_frame(frames):
    slot_values = []
    for frame in frames:
        for action in frame["actions"]:
            slot_values.append((action["slot"],action["values"][0]))
    return slot_values

def extract_dialogue_content(dialogues):
    dialogue_content = ""
    confirm_values = []
    utterance_id = 0
    for utterance_speaker in dialogues:
        utterance = utterance_speaker["utterance"]
        speaker = utterance_speaker["speaker"]
        for action in utterance_speaker["frames"][0]["actions"]:
            if action["act"].lower() == "confirm":
                utterance = "$$" + utterance_speaker["utterance"]
                confirm_values.append(extract_slot_values_from_frame(utterance_speaker["frames"]))
                break
        dialogue_content += speaker + ":" + utterance + "\n"
        utterance_id += 1
    return dialogue_content, confirm_values


def extract_key_utterance_content(dialogue_content):
    memory_related_utterance = []
    dialogue_content_split = dialogue_content.split("\n")
    for utterance in dialogue_content_split:
        if "$$" in utterance:
            memory_related_utterance.append(utterance.replace("$$",""))

    return "\n".join(memory_related_utterance)


if __name__ == '__main__':
    file_path = "/Users/duyiming/Downloads/TOD_MEM_CODE/dstc8/train/"
    # schemas = load_schema("/Users/duyiming/Downloads/TOD_MEM_CODE/dstc8/train/schema.json")
    exist_dialogue_id_file_path = 'raw_gpt4_0927_1211.json'
    existed_ids = read_generated_data(exist_dialogue_id_file_path)
    # load intents
    intents_samples, intents_in_sgd, sgd_intents_link = load_sgd_intents(file_path)
    output_data = {}
    all_index = 0
    generate_key_values = {}
    for domain_data in intents_samples.keys():
        index = 0
        for intent_dialogue in intents_samples[domain_data].keys():
            for dialogue_id, dialogue_content in intents_samples[domain_data][intent_dialogue].items():
                if dialogue_id in existed_ids:
                    continue
                if dialogue_id not in output_data.keys():
                    # intent 以及对应的dialogue
                    # try:
                    new_dialogue_contents = split_dialogue_content_by_intent(dialogue_content)
                    if len(new_dialogue_contents) < 3:
                        continue
                    if index > 2:
                        break
                    if all_index > 2:
                        break
                    # print(f'---------------{dialogue_id}----------------')
                    intent_dialogue_pure_text = {}
                    intent_dialogue_text = []
                    in_session_index = 0
                    total_confirm_values = []
                    for dialogue_contents in new_dialogue_contents:
                        for k, v in dialogue_contents.items():
                            text_content, confirm_values = extract_dialogue_content(v)
                            for confirm_slots in confirm_values:
                                if len(confirm_slots) < 2:
                                    continue
                                generate_key_values[all_index] =  {"task_goal":confirm_slots, "intent":k}
                                generate_key_values[all_index]["reference_dialogue_id"] = dialogue_id
                                generate_key_values[all_index]["domain"] = domain_data
                                generate_key_values[all_index]["intent"] = k

                    # print("---------------------------------")
                    index = index + 1

                    all_index = all_index + 1

                    # except Exception as e:
                    #     print(e)
            print(all_index)
    


    output_session_data = {}
    output_id = 0
    generate_prompt = """Help me generate an English conversation under the {dialogue_intent} intent,
      where {task_goal}. The conversation should be between a user and an assistant, 
      and it should be split into {task_goal_length} sessions at different points in time, 
      with continuity and connection between the sessions and each session should not less than 6 turns.
      Additionally, the final session must include a assistant response containing a complete confirmation-type utterance before the user confirms, and this utterance should be marked with `is_confirmation` set to `True`.
      and the user must provide a final confirmation response at the end of the final session. 
      For all other sessions, the conversation should end with an assistant's polite declarative statement.       
      """    
    for k, sample_key_words in generate_key_values.items():
        try:
            output_session_data[str(output_id)] = {}
            task_goal_description = ""
            for task_slot_value in sample_key_words["task_goal"]:
                task_goal_description += task_slot_value[0] + " is " + task_slot_value[1] + ", "
            session_length = min(len(sample_key_words["task_goal"]), 3)
            user_prompt = generate_prompt.format(dialogue_intent = sample_key_words["intent"], task_goal = task_goal_description , task_goal_length = session_length)
            system_prompt = """You are dialogue generator assistant. The sessions should be clearly separated, and the conversation should be formatted as follows:
                                Each turn should be a dictionary entry.
                                The conversation should be in the format of a list of sessions, where each session is a list of dictionaries representing each turn.
                                Each dictionary entry should have two keys: speaker (either 'user' or 'assistant') and text (the spoken dialogue).
                                Except for final session，each session should be a seperate dialogue and include a complete dialogue structure, beginning with a greeting from the user and ending with an assistant's polite declarative statement.
                                Feel free to expand the dialogue with additional relevant details, but avoid redundant expressions or repeating the same phrases.
                                Reponse me with a json format 
                                {"sessions": [
                                    [
                                        {
                                            "speaker": "xx",
                                            "text": "xx"
                                        },
                                    ],
                                    [
                                        {
                                            "speaker": "xx",
                                            "text": "xx"
                                        },
                                    ],
                                
                                ]}.
                            """
            response = llm_response(user_prompt, system_prompt)
            output_session_data[str(output_id)]["domain"] = sample_key_words["domain"]
            output_session_data[str(output_id)]["sessions"] = parse_response_results(response, sample_key_words)
            print(task_goal_description)
            print(response)
            output_id += 1
        except Exception as e:
            print(e)
    
    with open('raw_gpt4_0924_new_2.json', 'w') as json_file:
        json.dump(output_session_data, json_file, indent=4)
        print("JSON文件已成功保存")
    # 生成样本数据，核心函数。

