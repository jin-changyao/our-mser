import argparse
import pickle
import json
import os 
import pandas as pd
import matplotlib.pyplot as plt



def process_dataset(dataset, window=110, audio_description='True', audio_impression='False', audio_only='False', audio_context='False',experiments_setting='lora'):
    '''
    dataset: parameter that define the evaluated dataset.
    window:       parameter that control the historical context window
    speaker_mode: parameter that control the speaker identification task whether to be added in the main task

    data_path:    parameter that record the processed dataset's filepath
    '''
    label_set = {
        'iemocap':['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
        'meld':   ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
    }
    label_text_set = {
        'iemocap':'happy, sad, neutral, angry, excited, frustrated',
        'meld'   :'neutral, surprise, fear, sad, joyful, disgust, angry',
    }

    ## load audio feature files
    if dataset == 'meld':
        audio_feature_train = pd.read_csv('../speech_features/meld_processed_5class_train_audio_features.csv')
        audio_feature_test = pd.read_csv('../speech_features/meld_processed_5class_test_audio_features.csv')
        audio_feature_dev = pd.read_csv('../speech_features/meld_processed_5class_dev_audio_features.csv')
    elif dataset == 'iemocap':
        audio_feature = pd.read_csv('../speech_features/processed_iemocap_audio_features_5.csv')
        iemocap_file = pd.read_csv('../data/IEMOCAP_full_release/iemocap_full_dataset.csv')
    ###

    emotional_dict = {text_label:num_label for num_label, text_label in enumerate(label_set[dataset])}
    content_target_dict = {}
    content_task_dict = {}
    speaker_label_dict = {}
    sentence_dict = {}
    audio_path_dict = {}
    length = []
    data = pickle.load(open(f'../original_data/{dataset}/{dataset}.pkl','rb'))

    if dataset == 'iemocap':
        all_conv_id = data[3] + data[4] + data[5]
        sentence_dict = data[2]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                if speaker_label == 'M':
                    temp_speaker_list.append(0)
                else:
                    temp_speaker_list.append(1)
            speaker_label_dict[conv_id] = temp_speaker_list

    elif dataset == 'meld':
        all_conv_id = data[4] + data[5] + data[6]
        sentence_dict = data[3]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                temp_speaker_list.append(speaker_label.index(1))
            speaker_label_dict[conv_id] = temp_speaker_list

    # Process the utterances in the conversation, where 'index_w' is used to handle the starting index under the window size setting.
    
    for conv_id in all_conv_id:
        if conv_id==125:
            continue
        temp_conv_turn = 0
        for conv_turn in range(len(sentence_dict[conv_id])):
            temp_content_str = 'Now you are expert of sentiment and emotional analysis.'
            temp_content_str += 'The following conversation noted between \'### ###\' involves several speakers. '

            if audio_context == 'True':
                temp_content_str += 'The last three utterances are followed by its speech features. ### '
            else:
                temp_content_str += ' ### '

            index_w = max(conv_turn-window, 0)
            for i, (speaker_label, sub_sent) in enumerate(zip(speaker_label_dict[conv_id][index_w:conv_turn+1], sentence_dict[conv_id][index_w:conv_turn+1])):
                sub_sent = sub_sent.replace('', "'")
                sub_sent = sub_sent.replace('', "")
                sub_sent = sub_sent.replace('', "")
                sub_sent = sub_sent.replace('', "")
                sub_sent = sub_sent.replace('', " ")
                sub_sent = sub_sent.replace('', " ")
                temp_content_str += (f'\t Speaker_{speaker_label}:"{sub_sent}"')
                if audio_context == 'True':
                    if index_w+i<conv_turn-3:
                        continue
                    if dataset == 'iemocap':
                        filtered_rows = audio_feature.loc[(audio_feature['video_id'] == conv_id) & (audio_feature['text'] == sub_sent)]
                        avg_intensity = filtered_rows['avg_intensity_category'].values[0]
                        intensity_variation = filtered_rows['intensity_variation_category'].values[0]
                        avg_pitch = filtered_rows['avg_pitch_category'].values[0]
                        pitch_variation = filtered_rows['pitch_std_category'].values[0]
                        description = filtered_rows['description'].values[0].split(': ')[1]
                        # temp_content_str += f' ({avg_intensity} volume with {intensity_variation} variation)'
                        # temp_content_str += f' ({description})'
                        temp_content_str += f' ({avg_pitch} pitch with {pitch_variation} variation)'
                    elif dataset == 'meld':
                        sub_sent = sub_sent.replace('   ', ' ').replace('  ', ' ').strip()
                        if conv_id >= 1153:
                            diag_id = conv_id - 1153
                            audio_feature_file = audio_feature_test
                        elif conv_id >= 1039:
                            diag_id = conv_id - 1039
                            audio_feature_file = audio_feature_dev
                        else:
                            diag_id = conv_id
                            audio_feature_file = audio_feature_train
                        audio_feature_file['Utterance'] = audio_feature_file['Utterance'].apply(lambda x: str(x).replace('   ', ' ').replace('  ', ' ').strip())
                        try:
                            filtered_rows = audio_feature_file.loc[(audio_feature_file['Utterance'] == sub_sent)]
                            avg_intensity = filtered_rows['avg_intensity_category'].values[0]
                        except:
                            break
                        avg_intensity = filtered_rows['avg_intensity_category'].values[0]
                        intensity_variation = filtered_rows['intensity_variation_category'].values[0]
                        avg_pitch = filtered_rows['avg_pitch_category'].values[0]
                        pitch_variation = filtered_rows['pitch_std_category'].values[0]
                        description = filtered_rows['description'].values[0].split(': ')[1]
                        #temp_content_str += f' ({avg_intensity} volume with {intensity_variation} variation)'
                        #temp_content_str += f' ({description})'
                        temp_content_str += f' ({avg_pitch} pitch with {pitch_variation} variation)'
            content_target_dict[f'{conv_id}_{conv_turn}'] = label_set[dataset][data[1][conv_id][conv_turn]]
            target_utterance = temp_content_str.split('\t')[-1].split(' (')[0]
            temp_content_str += ' ###'
            
            if audio_only == 'True':
                temp_content_str = 'Now you are expert of sentiment and emotional analysis using only audio features.'

            # add audio feature and impression
            # if audio_description == 'True' or audio_impression == 'True':
            if dataset == 'meld':
                mode = ''
                if conv_id >= 1153:
                    diag_id = conv_id - 1153
                    audio_feature_file = audio_feature_test
                    mode = 'test'
                elif conv_id >= 1039:
                    diag_id = conv_id - 1039
                    audio_feature_file = audio_feature_dev
                    mode = 'dev'
                else:
                    diag_id = conv_id
                    audio_feature_file = audio_feature_train
                    mode = 'train'
                
                audio_directory = f'../data/MELD/{mode}/{mode}_audio'
                
                while temp_conv_turn<100:
                    filename = f'dia{diag_id}_utt{temp_conv_turn}.wav'
                    filtered_rows = audio_feature_file.loc[audio_feature_file['filename'] == filename]
                    if not filtered_rows.empty:
                        # get audio features
                        description = filtered_rows['description'].values[0]
                        impression = filtered_rows['impression'].values[0]
                        duration = filtered_rows['duration'].values[0]
                        # find audio path
                        path = audio_directory + "/" + filename
                        audio_path_dict[f'{conv_id}_{conv_turn}'] = path
                        # if conv_id==1149:
                        #     print(temp_conv_turn,path, conv_turn, target_utterance)
                        temp_conv_turn += 1
                        break
                    else:
                        temp_conv_turn += 1
                
                if duration > 0.5:
                    if audio_description == 'True':
                        temp_content_str += f' {description}'
                    if audio_impression == 'True':
                        temp_content_str += f' {impression}'
            elif dataset == 'iemocap':
                target_sentence = target_utterance.split(':')[1][1:-1]
                filtered_rows = audio_feature.loc[(audio_feature['video_id'] == conv_id) & (audio_feature['text'] == target_sentence)]
                description = filtered_rows['description'].values[0]
                impression = filtered_rows['impression'].values[0]
                audio_path = iemocap_file.loc[(audio_feature['video_id'] == conv_id) & (audio_feature['text'] == target_sentence)]['path'].values[0]
                audio_directory = '../data/IEMOCAP_full_release'
                audio_path_dict[f'{conv_id}_{conv_turn}'] = audio_directory + '/' + audio_path
                if audio_description == 'True':
                    temp_content_str += f' {description}'
                if audio_impression == 'True':
                    temp_content_str += f' {impression}'
            ##-------------------------------------------------------
            if experiments_setting != 'zero_shot':
                if audio_only == 'True':
                    temp_content_str += f' Please select the emotional label from <{label_text_set[dataset]}> based on the audio features. Respond with just one label:'
                elif audio_description == 'True' or audio_impression == 'True':
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> from <{label_text_set[dataset]}> based on both the context and audio features. Respond with just one label:'
                else:
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> from <{label_text_set[dataset]}> based on the context. Respond with just one label:'
            else:
                if audio_only == 'True':
                    temp_content_str += f' Please select the emotional label based on the audio features. Please ONLY output only one label from <{label_text_set[dataset]}> without any other words: '
                elif audio_description == 'True' or audio_impression == 'True':
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> based on both the context and audio features. Please output ONLY ONE label from <{label_text_set[dataset]}> as the first word without any other words: '
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> based on both the context and audio features. Please output ONLY ONE label from <{label_text_set[dataset]}> as the first word without any other words: '
                else:
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> based on the context. Please output ONLY ONE label from <{label_text_set[dataset]}> as the first word without any other words: '
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> based on the context. Please output ONLY ONE label from <{label_text_set[dataset]}> as the first word without any other words:'
            length.append(len(temp_content_str))
            
            content_task_dict[f'{conv_id}_{conv_turn}'] = temp_content_str


    if dataset == 'iemocap':
        train_ids, test_ids, valid_ids = data[3], data[4], data[5]
    elif dataset == 'meld':
        train_ids, test_ids, valid_ids = data[4], data[5], data[6]


    new_train_id, new_test_id, new_valid_id = [], [], []
    # new_train_target, new_test_target, new_valid_target = [], [], []
    for train_id in train_ids:
        for conv_turn in range(len(sentence_dict[train_id])):
            new_train_id.append(f'{train_id}_{conv_turn}')
            
        
    for test_id in test_ids:
        for conv_turn in range(len(sentence_dict[test_id])):
            new_test_id.append(f'{test_id}_{conv_turn}')
    
    for valid_id in valid_ids:
        for conv_turn in range(len(sentence_dict[valid_id])):
            new_valid_id.append(f'{valid_id}_{conv_turn}')

    # dataset_list = ['train', 'test', 'valid']
    data_path = f'../PROCESSED_DATASET/{dataset}/window/{audio_description}_{audio_impression}'
    os.makedirs(data_path, exist_ok=True)

    with open(f'{data_path}/train.json', 'w') as f_train:
        for train_id in new_train_id:
            if train_id.split('_')[0]=='125':
                continue
            f_train.write(json.dumps({'path': audio_path_dict[train_id], \
            'input':f'{content_task_dict[train_id]}','target':f'{content_target_dict[train_id]}'}, ensure_ascii=False)+ '\n')

    with open(f'{data_path}/test.json', 'w') as f_test:
        for test_id in new_test_id:
            f_test.write(json.dumps({'path': audio_path_dict[test_id],\
            'input':f'{content_task_dict[test_id]}','target':f'{content_target_dict[test_id]}'}, ensure_ascii=False)+ '\n')

    with open(f'{data_path}/valid.json', 'w') as f_valid:
        for valid_id in new_valid_id:
            if valid_id.split('_')[0]=='1149':
                continue
            f_valid.write(json.dumps({'path': audio_path_dict[valid_id], \
            'input':f'{content_task_dict[valid_id]}','target':f'{content_target_dict[valid_id]}'}, ensure_ascii=False)+ '\n')
    
        
    # draw histogram of the length of the input text
    plt.hist(length, bins=20)
    plt.xlabel('Length of the input text')
    plt.ylabel('Frequency')
    plt.title('Histogram of the length of the input text')
    plt.savefig(f'{data_path}/histogram.png')

    return data_path


parser = argparse.ArgumentParser(description='Data processing script')
parser.add_argument('--dataset', type=str, default='meld', help='Dataset name or path')
parser.add_argument('--historical_window', type=int, default=12, help='Historical window size')
parser.add_argument('--audio_description', type=str, default='False', help='Audio description task type')
parser.add_argument('--audio_impression', type=str, default='False', help='Audio impression task type')
parser.add_argument('--audio_only', type=str, default='False', help='Audio only task type')
parser.add_argument('--audio_context', type=str, default='False', help='Audio context task type')
parser.add_argument('--experiments_setting', type=str, default='lora', help='Experiments setting type')
args = parser.parse_args()




# Process data
processed_data_path = process_dataset(dataset=args.dataset, window=args.historical_window
        , audio_description=args.audio_description, audio_impression=args.audio_impression, 
        audio_only=args.audio_only, audio_context=args.audio_context, experiments_setting=args.experiments_setting)

print(processed_data_path)

