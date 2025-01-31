import copy
import torch
from scipy.fftpack import idct
import math
import sys
import DataTools
import csv

class V(object):
    def __init__(self):
        self.ADBmax = 0
        self.Reall2 = 0
        self.Reallinf = 0

bridge_bestKs = []

class Attacker():
    def __init__(self, args, model, im_orig, imgi, attack_method='TtBA',
                 tar_img=None, tar_label=None, dim_reduc_factor=4,
                 iteration=100, initial_query=30, tol=1e-4, sigma=3e-4):
        self.model = model
        self.x0 = im_orig.cuda()
        self.x0_label = model.predict_label(self.x0).cpu().item()
        self.out_label = self.x0_label
        self.tar_img = tar_img
        self.tar_label = tar_label
        if tar_img != None:
            self.tar_img = torch.unsqueeze(self.tar_img, dim=0)
        self.dim_reduc_factor = dim_reduc_factor
        if im_orig.shape[3] < 224:
            self.dim_reduc_factor = 1
        self.Max_iter = iteration
        self.Init_query = initial_query

        self.tol = tol
        self.sigma = sigma
        self.attack_method = attack_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.args = args
        self.Img_i = imgi
        self.old_best_adv = V()
        self.File_string = "TtBA"
        self.Img_result = []
        self.heatmaps = self.Img_result
        self.L2_line = [[0, self.x0.numel()**0.5]]
        self.Linf_line = [[0, 1.0]]
        self.success = -1
        self.queries = 0
        self.ACCquery = 0
        self.ACCiter_n = 1

    def generate_tensor_with_fixed_similarity(self, v, cos_sim):
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_unit = v / v_norm
        random_tensor = torch.randn_like(v, dtype=torch.float32).cuda()
        dot_product = torch.sum(random_tensor * v_unit, dim=-1, keepdim=True)
        projection = dot_product * v_unit
        perpendicular_vector = random_tensor - projection
        perpendicular_unit = perpendicular_vector / torch.norm(perpendicular_vector, dim=-1, keepdim=True)
        v2 = cos_sim * v_norm * v_unit + (1 - cos_sim ** 2) ** 0.5 * v_norm * perpendicular_unit
        return v2

    def tangent_vector_approximation(self, boundary_v, q_max):
        adv_label = self.model.predict_label(self.x0 + boundary_v).item()
        boundary_v_l2 = torch.norm(boundary_v).item()
        random_tangent_v = []
        labels_out = []
        z = []

        pixnum = torch.numel(self.x0)
        Cha_L22 = self.sigma ** 2 * pixnum
        SigmaTheta = 1 - Cha_L22 / (2 * boundary_v_l2 ** 2)
        for i in range(q_max):
            similar_v = self.generate_tensor_with_fixed_similarity(boundary_v, SigmaTheta)
            adv_v = similar_v - boundary_v
            noise_l2 = torch.norm(adv_v)
            labels_out.append(self.model.predict_label(similar_v + self.x0).item())
            random_tangent_v.append(adv_v)
            if self.tar_img == None:
                if labels_out[i] == self.x0_label:
                    z.append(-1)
                    random_tangent_v[i] *= -1
                else:
                    z.append(1)
            if self.tar_img != None:
                if labels_out[i] != self.tar_label:
                    z.append(-1)
                    random_tangent_v[i] *= -1
                else:
                    z.append(1)
            self.queries += 1
        normal_v = sum(random_tangent_v)
        mean_z = sum(z) / q_max
        return normal_v, mean_z

    def normal_vector_approximation(self, boundary_x, q_max):
        x0 = self.x0
        random_noises = None
        boundary_v = boundary_x - x0
        boundary_v_l2 = torch.norm(boundary_v).item()
        if self.dim_reduc_factor < 1.0:
            raise Exception(
                "The dimension reduction factor should be greater than 1 for reduced dimension, and should be 1 for Full dimensional image space.")
        if self.dim_reduc_factor > 1.0:
            fill_size = int(x0.shape[-1] / self.dim_reduc_factor)
            random_noises = torch.zeros(q_max, int(x0.shape[-3]), int(x0.shape[-2]), int(x0.shape[-1]),
                                        dtype=torch.float32).cuda()
            for i in range(q_max):
                random_noises[i][:, 0:fill_size, 0:fill_size] = torch.randn(x0.shape[0], x0.shape[1], fill_size,
                                                                            fill_size)
                random_noises[i] = torch.from_numpy(
                    idct(idct(random_noises[i].cpu().numpy(), axis=2, norm='ortho'), axis=1, norm='ortho'))
        else:
            #random_noises = torch.randn(q_max, x0.shape[1], x0.shape[2], x0.shape[3], dtype=torch.float32).cuda()
            #random_noises = torch.randint(0, 2, [q_max, x0.shape[1], x0.shape[2], x0.shape[3]], dtype=torch.float32).cuda() * 2 - 1
            random_noises = torch.rand(q_max, x0.shape[1], x0.shape[2], x0.shape[3], dtype=torch.float32).cuda() * 2 - 1
        labels_out = []
        pixnum = torch.numel(x0)
        for i in range(q_max):
            k_to_one = (math.sqrt(pixnum) / (self.dim_reduc_factor * torch.norm(random_noises[i]).item()))
            random_noises[i] *= (k_to_one * self.sigma)
            noise_l2 = torch.norm(boundary_v + random_noises[i])
            #if noise_l2 < boundary_v_l2:
            #    random_noises[i] *= -1
            labels_out.append(self.model.predict_label(x0 + boundary_v + random_noises[i]))
        self.queries += q_max

        z = []  # sign of grad_tmp
        for i, predict_label in enumerate(labels_out):
            if self.tar_img == None:
                if predict_label == self.x0_label:
                    z.append(-1)
                    random_noises[i] *= -1
                else:
                    z.append(1)
            if self.tar_img != None:
                if predict_label != self.tar_label:
                    z.append(-1)
                    random_noises[i] *= -1
                else:
                    z.append(1)
        normal_v = sum(random_noises)
        mean_z = sum(z) / q_max
        return normal_v, mean_z

    def get_attempt_unit_v(self, boundary_x, boundary_vl2, normal_v, k):
        normal_v /= torch.norm(normal_v)
        boundary_v = boundary_x[1] - self.x0
        boundary_v /= torch.norm(boundary_v)

        attempt_v = normal_v * (k) + boundary_v * (1 - k)
        attempt_v = attempt_v / torch.norm(attempt_v)
        #attempt_unit_x
        return attempt_v

    def TtBA(self, normal_v, boundary_x, boundary_vl2):
        basic_l2 = [0, copy.deepcopy(boundary_vl2[1].item())]
        GOLD = (5 ** 0.5 - 1) / 2
        num_q = 0
        L, R, mid = 0.0, 1.0, 0
        while abs(R-L) > 1e-3:
            mid = (L + R) / 2
            flag = self.is_adversarial(
                self.x0 + boundary_vl2[1] * self.get_attempt_unit_v(boundary_x, boundary_vl2, normal_v, mid))
            num_q += 1
            if flag == -1:
                R = mid
            else:
                L = mid

        low, JUMP, high, DIVE = 0.10, 0.9, 0.20, 2/3
        kk = (DIVE * high - JUMP * low) / (high - low)
        if L <= low:
            k = JUMP * L
        elif L >= high:
            k = DIVE * L
        else:
            k = kk * (L-low) + JUMP*low
            """"""
        #k = (2/3) * L
        attempt_unit_v = self.get_attempt_unit_v(boundary_x, boundary_vl2, normal_v, k)
        best_boundary_x, real_L2s, q_bin = self.bin_search_fast(
            torch.cat([self.x0, self.x0 + (boundary_vl2[1]) * attempt_unit_v]), self.tol)
        sim_cos = DataTools.cosine_similarity(best_boundary_x - self.x0, boundary_x - self.x0)
        return best_boundary_x, real_L2s, (num_q + q_bin), k, L


    def attack(self):
        if self.tar_img != None:
            x_random, query_random = self.tar_img.clone(), 0
        if self.tar_img == None:
            x_random, query_random = self.find_init_adv_x(self.x0)
        current_best_adv_x, current_best_adv_vl2, query_bin = self.bin_search_fast(
            torch.cat((self.x0.clone(), x_random.clone()), dim=0), self.tol)
        current_best_adv_v = current_best_adv_x - self.x0

        sys.stdout.write(
            f'\rImg{self.Img_i} query{self.queries :.0f} \tIter{0 :.0f} \treal_d={torch.norm(current_best_adv_v[1]):.6f}')
        sys.stdout.flush()

        total_ratios = []
        total_L = []
        for i in range(self.Max_iter):
            q_norm_v = int(self.Init_query * (i + 1) ** 0.5)
            # mid_l2k=sum(current_best_vl2)/(2*current_best_vl2[1])
            normal_v, ratios = self.normal_vector_approximation(current_best_adv_x[1], q_norm_v)
            #normal_v, ratios = self.tangent_vector_approximation(current_best_adv_v, q_norm_v)
            total_ratios.append(ratios)

            if self.attack_method == "TtBA":
                current_best_adv_x, current_best_adv_vl2, qs, k_app, L = self.TtBA(normal_v.clone(),
                                                                                     current_best_adv_x.clone(),
                                                                                     current_best_adv_vl2.clone())
            else:
                current_best_adv_x, current_best_adv_vl2, qs, k_best, L = self.TtBA_simp(normal_v.clone(),
                                                                                          current_best_adv_x.clone(),
                                                                                          current_best_adv_vl2.clone())
                """"""
                if 0.1 < L < 0.9 and i >= 2:
                    bridge_bestK = k_best / L
                    if 0.05 <= bridge_bestK:
                        bridge_bestKs.append(bridge_bestK)
            current_best_adv_v = current_best_adv_x[1] - self.x0
            total_L.append(L)
            norm_linf = torch.max(torch.abs(current_best_adv_v))

            sys.stdout.write(
                f'\rImg{self.Img_i} Que{self.queries :.0f} \tIter{i + 1:.0f} \tdist={current_best_adv_vl2[1]:.4f}, L={L:.3f} NVr={ratios:.3f}')
            sys.stdout.flush()

            self.old_best_adv.Reall2 = min(torch.norm(self.x0).item(), current_best_adv_vl2[1].item())
            self.old_best_adv.Reallinf = min(1, norm_linf.item())
            self.old_best_adv.ADBmax = min(torch.norm(self.x0).item(), current_best_adv_vl2[1].item())
            self.L2_line.append([self.queries, self.old_best_adv.Reall2])
            self.Linf_line.append([self.queries, self.old_best_adv.Reallinf])
            if self.queries >= self.args.budget:
                break
            if self.success == -1 and self.old_best_adv.Reall2 <= self.args.epsilon:
                self.success = 1
                self.ACCquery, self.ACCiter_n = self.queries, i
            if self.args.early == 1 and self.success == 1:
                break

        self.out_label = self.model.predict_label(torch.clamp(current_best_adv_x[1],0.0,1.0)).cpu().item()
        advimg = 0.5 * (1.0 + current_best_adv_v / self.old_best_adv.Reallinf)
        self.Img_result = [advimg, self.x0, self.x0 + current_best_adv_v]
        self.heatmaps = copy.deepcopy(self.Img_result)
        if self.tar_img is not None:
            self.heatmaps[0] = self.tar_img.clone()
            self.heatmaps[1] = 0.5 * (1.0 + self.tar_img)
        self.File_string = ("Img" + str(self.Img_i) + ",I-Q[" + str(self.ACCiter_n) + "-" +
                            str(self.queries) + "], ADB{:.4f}".format(self.old_best_adv.ADBmax) +
                            ",LB[" + str(self.x0_label) + "-" + str(self.out_label) + "]")
        print(f' --AVG_L={sum(total_L) / len(total_L):.3f}, AVG_ra={sum(total_ratios) / len(total_ratios):.3f}',
              end="")
        return bridge_bestKs

    def is_adversarial(self, image):
        predict_label = self.model.predict_label(torch.clamp(image,0.0,1.0)).cpu().item()
        self.queries += 1
        if self.tar_img == None:
            is_adv = predict_label != self.x0_label
        else:
            is_adv = predict_label == self.tar_label
        if is_adv:
            return 1
        else:
            return -1

    def find_init_adv_x(self, image):
        num_calls = 1
        step = 0.02
        perturbed = image
        while self.is_adversarial(perturbed) == -1:
            if self.args.budget / 10 <= num_calls <= self.args.budget / 10 + 2:
                perturbed = torch.ones(image.shape).to(self.device)
                if self.is_adversarial(perturbed) == -1:
                    perturbed = torch.zeros(image.shape).to(self.device)
            else:
                pert = torch.randn(image.shape).cuda()
                perturbed = torch.clamp(image + num_calls * step * pert, 0.0, 1.0)
                perturbed = perturbed.to(self.device)
                num_calls += 1
            sys.stdout.write(f'\rImg{self.Img_i} query{self.queries :.0f} \treal_d={torch.norm(perturbed):.6f} ')
            sys.stdout.flush()
        return perturbed, num_calls

    def bin_search_fast(self, adv_x, tol):
        out_adv_x = adv_x.clone()
        num_calls = 1
        l2 = torch.norm((out_adv_x - self.x0).view(2, -1), dim=1)
        while l2[1] - l2[0] > tol:
            num_calls += 1
            adv_mid = (out_adv_x[1] + out_adv_x[0]) / 2
            if self.is_adversarial(adv_mid) == 1:
                out_adv_x[1] = adv_mid
            else:
                out_adv_x[0] = adv_mid
            l2 = torch.norm((out_adv_x - self.x0).view(2, -1), dim=1)
        return out_adv_x, l2, num_calls


    def TtBA_simp(self, normal_v, boundary_x, boundary_vl2):
        L, R, mid = 0.0, 1.0, 0
        while abs(R-L) > 1e-3:
            mid = (L + R) / 2
            flag = self.is_adversarial(self.x0 + boundary_vl2[1] * self.get_attempt_unit_v(boundary_x, boundary_vl2, normal_v, mid))
            if flag == -1:
                R = mid
            else:
                L = mid
        low, JUMP, high, DIVE = 0.1, 0.9, 0.2, 2/3
        kk = (DIVE * high - JUMP * low) / (high - low)
        if L <= low:
            k_app = JUMP * L
        elif L >= high:
            k_app = DIVE * L
        else:
            k_app = kk * (L-low) + JUMP*low
        attempt_unit_v = self.get_attempt_unit_v(boundary_x, boundary_vl2, normal_v, k_app)
        best_boundary_x, real_L2s, q_bin = self.bin_search_fast(torch.cat([self.x0, self.x0 + (boundary_vl2[1]) * attempt_unit_v]), self.tol)

        num_q = 0
        k = 0.0
        ks = [k]
        step = 0.010
        best_i = 0
        attempt_boundary_x = [boundary_x]
        best_l2 = boundary_vl2.clone()
        attempt_boundary_vl2 = [best_l2.clone()]
        while k < 1.0:  # flag == -1:
            k += step
            ks.append(k)
            if k <= L:
                attempt_adv_x = self.x0 + boundary_vl2[1] * self.get_attempt_unit_v(boundary_x, boundary_vl2, normal_v, k)
                attempt_boundary_xnew, attempt_boundary_newvl2, query = self.bin_search_fast(
                    torch.cat((self.x0.clone(), attempt_adv_x.clone()), dim=0), 1e-4)
                attempt_boundary_x.append(attempt_boundary_xnew)
                attempt_boundary_vl2.append(attempt_boundary_newvl2)
                num_q += query
                if attempt_boundary_vl2[-1][1] < best_l2[1]:
                    best_i = len(attempt_boundary_vl2) - 1
                    best_l2 = attempt_boundary_vl2[-1]
            else:
                attempt_boundary_vl2.append(boundary_vl2)
            #else:
            #    break
        view = torch.stack(attempt_boundary_vl2)[:, 1].cpu()
        viewlist = view.tolist()
        with open("results_record/DecisionBoundaryLineVGGtar2.csv", encoding='utf-8', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(viewlist)
        #cos_sim = DataTools.cosine_similarity(attempt_boundary_x[best_i][1] - self.x0, boundary_x[1] - self.x0)
        #return attempt_boundary_x[best_i], attempt_boundary_vl2[best_i], num_q, best_i * step, L
        return best_boundary_x, real_L2s, num_q, best_i * step, L