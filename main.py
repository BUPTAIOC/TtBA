# coding:utf-8
import DataTools
import TtBA
import os
from datetime import datetime
import argparse
import line_profiler as lp
import torch
import random
import dataset

bridge_bestKs = []
def main():
    profile = lp.LineProfiler()
    # ###################################################################################
    torch_model, test_loader = None, None
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--dataset', default='mnist', type=str, help='Dataset')
    parser.add_argument('--targeted', default=0, type=int, help='targeted-1 or untargeted-0')
    parser.add_argument('--norm', default='k', type=str, help='Norm for attack, k or l2')
    parser.add_argument('--epsilon', default=0.3, type=float, help='attack strength')
    parser.add_argument('--budget', default=10000, type=int, help='Maximum query for the attack norm k')
    parser.add_argument('--early', default=0, type=int, help='early stopping (stop attack once the adversarial example is found)')
    parser.add_argument('--remember', default=0, type=int, help='if remember adversarial examples.')
    parser.add_argument('--imgnum', default=1000, type=int, help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int, help='begin test img number')
    parser.add_argument('--RGB', default=['RGB'], type=str, help='List of RGB channels (e.g., RG)')
    parser.add_argument('--binaryM', default=1, type=int, help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--initDir', default=1, type=int, help='initial direction, 1,-1,and 0 for random, 2 for 1-11-1...')

    args = parser.parse_args()
    print(args)

    Folder_name =  str(args.dataset) + "_" + str(args.norm) + "_aimR" + str(args.epsilon) +\
                        "_target" + str(args.targeted) +\
                        "_budget" + str(args.budget) +\
                        "_md" + str(args.binaryM) +\
                        "_Early" + str(args.early) +\
                        "_IMG" + str(args.beginIMG) + "+" + str(args.imgnum) +\
                        "_T" + str(datetime.now().strftime("%H-%M-%S"))
    if not os.path.exists("results_record/" + Folder_name):
        os.makedirs("results_record/" + Folder_name)

    # ###############################################################################
    test_loader, torch_model = dataset.load_dataset_model(args.dataset)
    Out = dataset.OutResult(args, Folder_name)

    Attacker = None
    for imgi, (original_image, label) in enumerate(test_loader):
        if Out.ImgNum_origin_right >= args.imgnum:
            break
        if imgi < args.beginIMG:
            continue
        Out.ImgNum_total_tested = Out.ImgNum_total_tested + 1
        out_label = torch_model.predict_label(original_image.cuda()).cpu().item()
        real_label = label.item()
        if out_label == real_label:
            ATK_target_xi, ATK_target_yi = None, None
            if args.targeted == 1:
                random_index = random.randint(0, len(test_loader) - 1)
                ATK_target_xi, ATK_target_yi = test_loader.dataset[random_index]
                ATK_target_xi = ATK_target_xi.cuda()
                ATK_target_F = torch_model.predict_label(ATK_target_xi).cpu()
                while label.item() == ATK_target_yi or ATK_target_F.item() != ATK_target_yi:
                    random_index = random.randint(0, len(test_loader) - 1)
                    ATK_target_xi, ATK_target_yi = test_loader.dataset[random_index]
                    ATK_target_xi = ATK_target_xi.cuda()
                    ATK_target_F = torch_model.predict_label(ATK_target_xi).cpu()

            if args.norm == "TtBA":
                Attacker = TtBA.Attacker(args, torch_model, original_image, imgi, args.norm, ATK_target_xi, ATK_target_yi)
                Attacker.attack()
            else:
                print("norm is wrong: "+args.norm)
                return

            Out.add1Result(Attacker)
            Out.Summary()
            if args.remember == 1:
                combined_file = DataTools.save_images([Attacker.Img_result, Attacker.heatmaps], "results_record/" + Folder_name, Attacker.File_string)
            print("")
        else:
            print(f"IMG{imgi} Originally classify incorrect")
    Out.Summary()
    print(args)
    print(f"NATURAL ACCURACY RATE={Out.NATURAL_ACCURACY_RATE:.4f}")
    print(f"ATTACK SUCCESS RATE = {Out.ATTACK_SUCCESS_RATE:.4f}")
    print(f"ROBUST ACCURACY RATE={Out.ROBUST_ACCURACY_RATE:.4f}")
    print(f"AVG(MID)-AccQuery  = {Out.AccQuery_avg:.1f}({Out.AccQuery_mid:.1f})")
    print(f"midAUC-l2(linf) = {Out.AUC_l2:.1f}({Out.AUC_linf:.1f})")
    query = [int(args.budget/5), int(args.budget/2), int(args.budget)]
    for q in query:
        print(f"AVG(MID)-l2 after {q} queries : {torch.mean(Out.L2_LINE_sum[:, q]).item():.3f}"
              f"({torch.median(Out.L2_LINE_sum[:, q]).item():.3f})")
    #print(f"AVG(MID)-linf= {Out.Endlinf_avg:.3f}({Out.Endlinf_mid:.3f})")
    """"""
    print(f"AVG(MID)-ADB = {Out.EndADB_avg}({Out.EndADB_mid})")
    print(f"AVG(MID)-linf= {Out.Endlinf_avg}()")


if __name__ == "__main__":
    main()