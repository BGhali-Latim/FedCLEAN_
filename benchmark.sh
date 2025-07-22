#!/bin/bash

echo "started script" >> progress.out

experiment='Backdoor'
############################################# Baseline#########################################
#
#for dataset in MNIST; do
#    for algo in noDefense; do 
#       for attack in NoAttack SignFlip SameValue AdditiveNoise Scaled; do  
#         # python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling CAM | tee "Results/$algo.$dataset.$attack.CAM"
#         # echo "$dataset.$algo.$attack.CAM" >> progress.out
#         python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling Dirichlet_per_class | tee "Results/$algo.$dataset.$attack.Dirichlet"
#         echo "$dataset.$algo.$attack.Dirichlet" >> progress.out
#       done
#    done
#done
#echo "finished baseline" >> progress.out

############################################# Niid FedCAM #########################################

for dataset in MNIST; do
   for algo in fedCAM_dev; do 
      # for attack in NoAttack SignFlip SameValue AdditiveNoise Scaled; do 
      for attack in Neurotoxin; do  
        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling CAM | tee "Results/$algo.$dataset.$attack.CAM"
        echo "$dataset.$algo.$attack.CAM" >> progress.out
        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling Dirichlet_per_class | tee "Results/$algo.$dataset.$attack.Dirichlet"
        echo "$dataset.$algo.$attack.Dirichlet" >> progress.out
      done
   done
done
echo "finished ours" >> progress.out

############################################# FedCAM #########################################

#for dataset in MNIST; do
#   for algo in fedCAM; do 
#      for attack in NoAttack SignFlip SameValue AdditiveNoise Scaled; do  
#      #   python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling CAM | tee "Results/$algo.$dataset.$attack.CAM"
#      #   echo "$dataset.$algo.$attack.CAM" >> progress.out
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling Dirichlet_per_class | tee "Results/$algo.$dataset.$attack.Dirichlet"
#        echo "$dataset.$algo.$attack.Dirichlet" >> progress.out
#      done
#   done
#done
#echo "finished ours" >> progress.out

############################################# FLEDGE #########################################
#
#for dataset in MNIST FashionMNIST; do
#   for algo in fledge; do 
#      for attack in NoAttack SignFlip SameValue AdditiveNoise Scaled; do  
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling CAM | tee "Results/$algo.$dataset.$attack.CAM"
#        echo "$dataset.$algo.$attack.CAM" >> progress.out
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling Dirichlet_per_class | tee "Results/$algo.$dataset.$attack.Dirichlet"
#        echo "$dataset.$algo.$attack.Dirichlet" >> progress.out
#      done
#   done
#done
#echo "finished ours" >> progress.out
############################################ FedCVAE  #########################################
#
#for dataset in MNIST FashionMNIST; do
#   for algo in fedCVAE; do 
#      for attack in NoAttack SignFlip SameValue AdditiveNoise Scaled; do  
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling CAM | tee "Results/$algo.$dataset.$attack.CAM"
#        echo "$dataset.$algo.$attack.CAM" >> progress.out
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling Dirichlet_per_class | tee "Results/$algo.$dataset.$attack.Dirichlet"
#        echo "$dataset.$algo.$attack.Dirichlet" >> progress.out
#      done
#   done
#done
#echo "finished ours" >> progress.out
############################################ FedGuard #########################################
#
#for dataset in MNIST FashionMNIST; do
#   for algo in fedGuard; do 
#      for attack in NoAttack SignFlip SameValue AdditiveNoise Scaled; do  
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling CAM | tee "Results/$algo.$dataset.$attack.CAM"
#        echo "$dataset.$algo.$attack.CAM" >> progress.out
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling Dirichlet_per_class | tee "Results/$algo.$dataset.$attack.Dirichlet"
#        echo "$dataset.$algo.$attack.Dirichlet" >> progress.out
#      done
#   done
#done
#echo "finished ours" >> progress.out


# experiment='cnn_prod'
#
# dataset="MNIST" 
# for algo in fedCAM_dev; do 
#    for attack in NoAttack SignFlip SameValue AdditiveNoise; do 
    #    echo "running $algo non iid with 30% $attack attackers on $dataset" 
    #    python3 TestMain_alt.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment| tee ./logs/$experiment.$algo.$dataset.$attack.out
#    done
# done
# AdditiveNoise SignFlip SameValue

#Compare attacks
#cd ./Plots 
#for metric in test_accuracy; do 
#    for attack in AdditiveNoise SignFlip SameValue; do 
#        python3 generate_samples.py -mode several \
#        -baseline False \
#        -result_dir ../Final/1000-50/MNIST/ \
#        -experiment_list  "NoDefense/non-IID_30_NoAttack" "NoDefense/non-IID_30_$attack" "FedCAM/non-IID_30_$attack" \
#        "FedCVAE/non-IID_30_$attack" "FedGuard/non-IID_30_$attack"\
#        -metric "$metric""_100.json"
#        python3 plot_results_compare_smoothed.py \
#        -columns "Baseline" "NoDefense" "FedCAM" "FedCVAE" "FedGuard"\
#        -figname "$attack""_$metric""0.3_all_comparison" \
#        -figtitle "$metric comparison of algos in a $attack scenario"
#    done
#done
#
## Compare Noattack
##cd ./Plots 
#for metric in test_accuracy; do 
#    for attack in NoAttack; do 
#        python3 generate_samples.py -mode several \
#        -baseline False \
#        -result_dir ../Final/1000-50/MNIST/ \
#        -experiment_list  "NoDefense/non-IID_30_$attack" "FedCAM/non-IID_30_$attack" \
#        "FedCVAE/non-IID_30_$attack" "FedGuard/non-IID_30_$attack"\
#        -metric "$metric""_100.json"
#        python3 plot_results_compare_smoothed.py \
#        -columns "NoDefense" "FedCAM" "FedCVAE" "FedGuard"\
#        -figname "$attack""_$metric""0.3_all_comparison" \
#        -figtitle "$metric comparison of algos in a $attack scenario"
#    done
#done



