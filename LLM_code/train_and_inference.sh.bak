# The Shellparameter that controls the mainprocess
FLAG=1
# bash train_and_inference.sh
# Please adjust the following parameters according to your needs. Rememeber to update the MODELPATH for each LLM model.

# ------  select basemodel ----------
# MODEL_NAME='LLaMA2'
# MODEL_NAME='LLaMA3'
# MODEL_NAME='LLaMA3-instruct-70b'
MODEL_NAME='LLaMA3-instruct'
# MODEL_NAME='Phi3-medium'

# ------ select the experiment ------------
# Experiments_setting='test'
Experiments_setting='zero_shot'
# Experiments_setting='few_shot'
# Experiments_setting='lora'
# Experiments_setting='all_parameters'

#  ------ select the dataset ------ 
dataset='iemocap'
# dataset='meld'

# ------  prompt input format setting ------ 
audio_description='False'
audio_impression='False'
audio_context='False' # add audio description for the last three utterances of the context
audio_only='False' # do not use text input

# ------  training setting ------ 
SEED=11
num_train_epochs=15
LORA_LR=3e-4
# training setting for projection-based model
use_encoder='False' # use False for SpeechCueLLM, True for projection-based model
projector='linear' # projector: linear, q-former
freeze_llm='False'
freeze_encoder='True'
# set the accumulation and card when backwarding and inferring
accumulations=8
graphics_card=1
mini_batch_size=1
BS=$((accumulations * graphics_card * mini_batch_size))
# set the port for deepspeed: use different ports if running in parallel
port=26000
# name the experiment (your choice)
task='des_context'

#  ------ select the historical window for dataset ------ 
# LLaMA 's context = 1024 is enough for almost dataset, except for iemocap.
# IEMOCAP has very long conversation sample, 
# the historical window is designed for this kind of long conversation.
historical_window=12

data_percent=1.0
# -----------------------------------------------------------------------------




case ${MODEL_NAME} in
'ChatGLM'|'ChatGLM2'|'LLaMA'|'LLaMA2'|'LLaMA3'|'LLaMA3-instruct'|'LLaMA3-instruct-70b'|'Bloom-560m'|'Phi3-medium')
    case ${Experiments_setting} in
    'zero_shot'|'few_shot'|'lora'|'all_parameters')
        case ${dataset} in
        'iemocap'|'meld')
            echo "******************************************************************************************"
            echo "All parameters are valid."
            echo "The dataset you have selected is: ${dataset} !"
            echo "The base model you have selected is ${MODEL_NAME}!"
            echo "The model's SFT method you have selected: ${Experiments_setting}!"
            echo "******************************************************************************************"
            ;;
        *)
            echo "The dataset parameter is invalid. CHECK IT OUT!"
            FLAG=0
            ;;
        esac
        ;;
    *)
        echo "The Experiments_setting parameter is invalid. CHECK IT OUT!"
        FLAG=0
        ;;
    esac
    ;;
*)
    echo "The MODEL_NAME parameter is invalid. CHECK IT OUT!"
    FLAG=0
    ;;
esac


if [ ${FLAG} = 1 ]
then
    DATA_PATH=$(python data_process.py --dataset ${dataset} \
        --historical_window ${historical_window} \
        --audio_description ${audio_description} \
        --audio_impression ${audio_impression} \
        --audio_only ${audio_only}\
        --audio_context ${audio_context}\
        --experiments_setting ${Experiments_setting}) \


    if [ $? -eq 0 ]; then
        echo "******************************************************************************************"
        echo -e "Data procession has executed successfully !"
        echo "******************************************************************************************"

    else
        echo "Data procession script encountered an error."
    fi

    if [ ${dataset} = 'iemocap' ]    
    then
        # MAX_LENGTH=1200
        MAX_LENGTH=2500
    elif [ ${dataset} = 'meld' ]
    then
        #MAX_LENGTH=1024
        MAX_LENGTH=1500
    else
        echo -e "Your choose is not in MY candidations! Please check your Model name!"
    fi
    echo "******************************************************************************************"
    echo -e "Your choose ${dataset}! The max_context_length will be set as ${MAX_LENGTH}!"
    echo "******************************************************************************************"


    if [ ${MODEL_NAME} = 'ChatGLM' ]
    then
        MODEL_PATH='CHATGLM MODELPATH'
    elif [ ${MODEL_NAME} = 'LLaMA' ]
    then
        MODEL_PATH='LLaMA MODELPATH'
    elif [ ${MODEL_NAME} = 'LLaMA2' ]
    then
        MODEL_PATH='LLaMA2 MODELPATH'
    elif [ ${MODEL_NAME} = 'LLaMA3' ]
    then
        MODEL_PATH='LLaMA3 MODELPATH'
    elif [ ${MODEL_NAME} = 'LLaMA3-instruct' ]
    then
        MODEL_PATH='LLaMA3-instruct MODELPATH'
    elif [ ${MODEL_NAME} = 'LLaMA3-instruct-70b' ]
    then
        MODEL_PATH='LLaMA3-instruct-70b MODELPATH'
    elif [ ${MODEL_NAME} = 'Phi3-medium' ]    
    then
        MODEL_PATH='Phi3-medium MODELPATH'
    else
        echo -e "Your choose is not in MY candidations! Please check your Model name!"
    fi
    echo -e "Your choose ${MODEL_NAME}! Model Parameters should be initialized in the path \n ${MODEL_PATH}"


    if [ ${Experiments_setting} = 'zero_shot' ]
    then
        DO_EVAL=True
        DO_TRAIN=False
        LORA=False
        LR=0
        CHECKPOINT_DIR=None
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as ZERO_SHOT model"
    elif [ ${Experiments_setting} = 'few_shot' ]
    then
        DO_EVAL=True
        DO_TRAIN=False
        LORA=False
        LR=0
        CHECKPOINT_DIR=None
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as FEW_SHOT model"
    elif [ ${Experiments_setting} = 'lora' ]
    then
        DO_EVAL=True
        DO_TRAIN=True
        LORA=True
        LR=${LORA_LR}
        CHECKPOINT_DIR=None
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as LORA model"
    elif [ ${Experiments_setting} = 'all_parameters' ]
    then
        DO_EVAL=True
        DO_TRAIN=True
        LORA=False
        LR=2e-5
        CHECKPOINT_DIR=None
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as ALL_PARAMETERS model"
    else
        echo -e "Your choose is not in MY candidations! Please CHECK your Experiments Setting!"
    fi


    if [ ${use_encoder} = 'True' ]
    then
        echo "Processed Data_Path: $DATA_PATH"
        deepspeed --master_port=${port} main_encoder.py \
        --dataset ${dataset} \
        --model_name_or_path ${MODEL_PATH} \
        --data_dir ${DATA_PATH} \
        --output_dir ../experiments/${MODEL_NAME}/${Experiments_setting}/${dataset}/window_${historical_window}/LR_${LR}_BS_${BS}_per_${data_percent}_${task}_class5_${SEED} \
        --max_length ${MAX_LENGTH} \
        --batch_size ${BS} \
        --deepspeed_config ../LLM_code/data_utils/deepspeed_config.json \
        --gradient_accumulation_steps ${accumulations} \
        --eval_batch_size 8 \
        --num_train_epochs ${num_train_epochs} \
        --save_steps 100000 \
        --lora ${LORA}\
        --learning_rate ${LR} \
        --do_eval ${DO_EVAL} \
        --do_train ${DO_TRAIN} \
        --statistic_mode True \
        --data_percent ${data_percent} \
        --seed ${SEED} \
        --freeze_llm ${freeze_llm} \
        --freeze_encoder ${freeze_encoder}\
        --projector ${projector}
    else
        echo "Processed Data_Path: $DATA_PATH"
        deepspeed --master_port=${port} main.py \
        --dataset ${dataset} \
        --model_name_or_path ${MODEL_PATH} \
        --data_dir ${DATA_PATH} \
        --output_dir ../experiments/${MODEL_NAME}/${Experiments_setting}/${dataset}/window_${historical_window}/LR_${LR}_BS_${BS}_per_${data_percent}_${task}_class5_${SEED} \
        --max_length ${MAX_LENGTH} \
        --batch_size ${BS} \
        --deepspeed_config ../LLM_code/data_utils/deepspeed_config.json \
        --gradient_accumulation_steps ${accumulations} \
        --eval_batch_size 8 \
        --num_train_epochs ${num_train_epochs} \
        --save_steps 100000 \
        --lora ${LORA}\
        --learning_rate ${LR} \
        --do_eval ${DO_EVAL} \
        --do_train ${DO_TRAIN} \
        --statistic_mode True \
        --data_percent ${data_percent} \
        --seed ${SEED}
    fi  
fi