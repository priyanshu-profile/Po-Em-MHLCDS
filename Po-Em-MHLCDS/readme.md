The structure of the entire setup is as follows:

```
|___________/Codes
|	       |___ /MLE_Loss_Dialogue_Model
|	       		|___ MLE_Loss_Dialogue_model.py	# script to train mle-loss based counseling dialogue model.
|
|	       |___ /RL_Fine-Tuning
|	       		|___ dataset.py				# script to load the custom dataset by creation of a pytorch Dataset class
|	       		|___ rlmain.py				# script to fine-tune mle-loss based dialogue model.
|	       		|___ rlutils.py                     # script containing utility functions and reward functions for the RL fine-tuning task
|                       |___ ppo.py 				# script containing implementation of buffer memory
|                       |___ loss.py 				# script containing implementation of Sequence Cross Entropy Loss
|              		|___ rlinference.py 			# script to interact with the fine-tuned model.
|
|
|	       |___ /Classifiers
|                       |___ classifier.py 			# script to implement classifiers
|
|
|
|            |___ /Evaluation
|	       		|___ interact.py				# script to evaluate the proposed system.
|
|___________/Datasets	 				 	         
|              |___ MHLCD.csv						# Mental Health and Legal Counseling Dialogue Dataset with counseling, politeness and empathy information. 
```

****REQUIREMENTS****
1. numpy: version '1.21.2'
2. pandas: version '1.3.4'
3. transformers: version '4.11.2'
4. tqdm: version: version '4.62.3'
5. torch: version '1.10.0'


****FINE-TUNING RL MODEL****

1. Provide all the arguments in the "rlmain.py" file.
2. Go to terminal window and enter "python rlmain.py" for WindowsOS or "python3 rlmain.py" for UNIX-based OS to start the RL fine-tuning.

Args:

modelname:str, 'the desired modelname',
csvfile:str, the csv file to load the annotated dataset from
device:str, Default='cuda'
n_epochs:int, Default=1
batch_size:int, Default=1
mini_batch=int, Default=1
train_single_model:bool, Whether to fine-tune both persuader and persuadee or either one of them during RL fine tuning, Default=True
single_model_to_train:str, Which model of train 'persuader' or 'persuadee', Default:'persuader',
num_candidates:int, number of candidates to generate at a turn for the persuader, Default=3
recompute_log_prob:bool, Whether to recompute the log probability of the generated candidates, Default= True
average_sent_loss:bool, Whether to average the loss the over the entire sentence for the generated candidates, Default=True
max_candidate_length:int, Maximum length of generated candidates, Default=50
human_reward:int, Default=10
beta2:float, Default=2
beta3:float, Default=2
beta4:float, Default=2
top_p:float, The probability sum threshold to consider when generating tokens for the candidates,  Default=0.9
temperature:float, The temprate value when calculating the loss, Default=0.8
use_recent_past:bool, Whether to consider the recent past
warmup_steps:int, number of warm up step to be given to the scheduler, Default=10
print_every:int, number of steps before printing the loss Default=1
evaluate_every:int, Iterations before evaluation, Default=1
learning_rate:float, Default=2e-05
epsilon:float, Default=0.2
loadModel:bool, Whether to load the pretrained language model for fine-tuning, Default=True
loadFilename:str, path to the saved pretrained language model
pad_token_id:int, Default=2
seedvalue:int, Default=10
use_counseling_classifier:bool whether to use counseling classifier, Default=True
use_politeness_classifier:bool whether to use politeness classifier, Default=True
use_empathy_classifier:bool whether to use empathy classifier, Default=True
con_classifier_filename:str,  path to the saved counseling classifier
bin_classifier_filename:str, path to the saved binary classifier
pol_classifier_filename:str, path to the saved politeness classifier
emp_classifier_filename:str, path to the saved empathy classifier
con_num_labels:int, number of counseling labels, Default=11
pol_num_labels:int, number of politeness labels, Default=3
emp_num_labels:int, number of empathy labels, Default=2
use_jaccard:bool, Whether to use non-repetitiveness as reward, Default=True
use_context:bool, Whether to use contextual-coherence as a reward, Default=True
gamma1:float, weight for the counseling-strategy reward, Default=0.3
gamma2:float, weight for the politeness reward, Default=0.2
gamma3:float, weight for the empathy reward, Default=0.2
gamma4:float, weight for the contextual-coherence reward, Default=0.2
gamma5:float, weight for the jaccard reward, Default=0.1
