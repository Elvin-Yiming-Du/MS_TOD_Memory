
openai_model_config = {
    "config_name": "gpt-4o_config",
    "model_type": "openai_chat",
    "model_name": "gpt-4o-mini",
    "api_key": "xxx",
}

import json, sys, os
sys.path.append('agentscope')
from agentscope.agents import DialogAgent
import agentscope
from agentscope.message import Msg

import re
is_memory=False
agentscope.init(model_configs=openai_model_config)

# Create a dialog agent and a user agent
user_agent = DialogAgent(name="assistant",
                               model_config_name="gpt-4o_config",
                              sys_prompt="You're a user in a task-oriented dialogue.")
our_agent = DialogAgent(name="assistant",
                               model_config_name="gpt-4o_config",
                              sys_prompt='Get slot_list from user. Ask one missing slot at a time. Keep it natural. Stop when all are obtained.')

# user_agent = UserAgent()


def get_message(content, mem, name):
    message = Msg(
        name=name,
        content=content,
        role="user",
        mem = mem
    )
    # message['mem'] = mem
    return message

memory_dict = {
    "unacquired slot":"(yes or no)",
    "wanted slot":"",
    "response":""
}

x = None
root = 'clean_personal_dataset_0928_131'
dialog_dict = {}
people = 0
print(len(os.listdir(root)))
# exit()

with open('new_wo_mem.json','r',encoding='utf-8') as f:
    datas = json.load(f)
check_dict={}
for i in range(132):
    check_dict[str(i)]=0
for item, value in datas.items():
    # print(item,type(item))
    check_dict[item]=1

for file in os.listdir(root):
    # people += 1
    people = int(file.split('_')[-1].split('.')[0])
    if check_dict[str(people)]==1:
        continue
    path = os.path.join(root,file)
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    history_session = []
    dialog_tmp_dict = {}
    # current_session = []
    sessions = data['sessions']
    num = 0
    for sess in sessions:
        dialog = []
        content_list = sess['content']
        
        print(file,sess['task_goal'])
        if len(sess['task_goal']) != 0:
            print('here')
            task_goal_prompt = ''
            for goal in sess['task_goal']:
                content = ''
                
                slot_dict = {}
                slot_list = []
                for each in goal['slot_values']:
                    slot_dict[each[0]] = each[1]
                    slot_list.append(each[0])
                begin_question = ''
                current_content = ''
                current_content+=content_list[0]['speaker']+':'+content_list[0]['utterance']+'\n'

                if is_memory and len(history_session)!=0:
                    begin_question+='\n'.join(history_session)

                dialog.append({'content':current_content})
                turn = 0

                while True:
                    try:
                        turn += 1
                        if turn > 30:
                            break
                        sys_prompt = '[History dialog]:' + begin_question + \
                                     '\n[User utterance]:' + current_content + \
                                     '\nYou goal is to get the user\'s information including: ' + str(slot_list) + \
                                     '\nPlease 1. Are slots not acquired?' \
                                     '1. Determine the which slot you want to obtain' \
                                     '2. To get the slot information, use one sentence to reply [User utterance]. ' \
                                     'If user utterance is an order, please answer first than propose questions for slot information.' \
                                     '3. Fill the following json:\n.' + str(memory_dict)
                        print('=========our=========')
                        our_res, our_memory = our_agent(
                            get_message(sys_prompt, 'delete', 'system'))
                        user_prompt = '[Slot information]:' + str(slot_dict) + '\n[System utterance]:' + our_res[
                            'content'] + \
                                      '\nBased on the slot information, response to [System utterance] in one sentence.'
                        print('=========user=========')
                        user_res, user_memory = user_agent(
                            get_message(user_prompt, 'delete', 'user')
                        )

                        text = our_res['content']
                        print('text:', text)
                        if "'response'" in text:
                            before = text.split("'response'")[0]
                            text = text.split("'response'")[-1].replace('"', '').replace('}', '').replace(':', '')
                            text = before + "'response': '" + text + "'}"
                        elif '"response"' in text:
                            before = text.split('"response"')[0]
                            text = text.split('"response"')[-1].replace('"', '').replace('}', '').replace(':', '')
                            text = before + "'response': '" + text + "'}"

                        # print('text:',text)
                        start_index = text.find("{")
                        end_index = text.rfind("}") + 1
                        json_str = text[start_index:end_index].replace('\n', '') \
                            .replace("'unacquired slot': '", '"unacquired slot": "') \
                            .replace("'response': '", '"response": "') \
                            .replace("'wanted slot': '", '"wanted slot": "') \
                            .replace("'}", '"}') \
                            .replace("',", '",').replace('"Black Widow"', "'Black Widow'") \
                            .replace("'", '')

                        sys_answer = eval(json_str)

                    except Exception as e:
                        continue
                        dialog.append(text)
                        dialog.append({'content': user_res['content']})
                        print('wrong file:',e,file)
                        exit()
                    dialog.append(sys_answer)
                    dialog.append({'content': user_res['content']})

                    print('sys_prompt:', sys_prompt)
                    print('user_prompt:', user_prompt)
                    print('sys_answer:', sys_answer)
                    print('user_answer:', user_res['content'])
                    print('=================================')

                    if 'no' in sys_answer['unacquired slot'].lower():
                        break
                    tmp_dict = {
                        'utterance_id': goal['utterance_id'],
                        'session_id':sess['session_id'],
                        'content': dialog
                    }
                    dialog_tmp_dict[num]=tmp_dict#dialog
                    dialog_dict[people]=dialog_tmp_dict
                    
                    with open('new_wo_mem_2.json','w',encoding='utf-8') as f:
                        json.dump(dialog_dict,f,indent=4)

                    current_content = user_res['content']

                    begin_question += 'system:'+sys_answer['response']+'\n'
                    begin_question += 'user:' + current_content + '\n'


                num+=1

        for content in content_list:
            history_session.append(content['speaker']+':'+content['utterance'])

