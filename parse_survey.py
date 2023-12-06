import json
import os
import csv
from enum import Enum

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics

from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind


def main():
    
    sns.set_theme(style="dark", font="Courier New")
    sns.set_context("talk")
    plt.rcParams['font.weight'] = 'bold'

    survey_file = os.path.join("data", "emnlp2022_survey.json")
    with open(survey_file, 'r') as f:
        contents = f.read()

    parsed_contents = json.loads(contents)

    academic_papers = []
    non_profit_papers = []
    profit_papers = []
    all_papers = []

    for paper in parsed_contents["papers"]:
        a = get_affil(paper["affiliations"])
        if (a == Affil.ACADEMIC):
            academic_papers.append(paper)
        elif (a == Affil.NON_PROFIT):
            non_profit_papers.append(paper)
        elif (a == Affil.PROFIT):
            profit_papers.append(paper)
        all_papers.append(paper)

    groups = [PaperGroup(academic_papers, Affil.ACADEMIC), 
              PaperGroup(non_profit_papers, Affil.NON_PROFIT),
              PaperGroup(profit_papers, Affil.PROFIT)]

    all_papers = PaperGroup(all_papers)

    plot_box_single(groups, log_scale=False, save=True,
            save_name="box_combined_outliers_removed.pdf", 
            remove_negatives=False, remove_outliers=True, 
            thresh_scale=2, t_test=False)

    plot_box(groups, log_scale=False, save=True,
            save_name="box_split_outliers_removed.pdf", 
            remove_negatives=False, remove_outliers=True, 
            thresh_scale=2, t_test=False)

    plot_years(all_papers, smooth='gaussian', save=True,
             save_name="line_years.pdf")

    plot_stacked_bars(groups, title=None,
            save=True, save_name="stacked_bars.pdf")


class PaperGroup:
    def __init__(self, papers, affiliation=None):

        # num_long = 0
        # num_code = 0
        # for p in papers:
        #     if p["code"] == True:
        #         num_code += 1
        #     if p["long"] == True:
        #         num_long += 1
        # 
        # num_both = 0
        # for p in papers:
        #     if p["code"] == False and p["long"] == False:
        #         num_both += 1

        # print(len(papers), num_code, num_long, num_both)

        self.papers = papers
        self.num_papers = len(papers)

        self.affiliation = affiliation

        self.data = {
            "paper_types" : {},
            # Sometimes multiple types per paper
            "num_paper_types" : 0,
            # paper_types is sum, this dictionary lists the paper identifiers
            # for each group
            "ids_of_each_type" : {},

            "no_basemodel_affils" : 0,
            "basemodel_affils" : {
                Affil.ACADEMIC.name : 0,
                Affil.NON_PROFIT.name : 0,
                Affil.PROFIT.name : 0
            },
            "num_basemodel_affils" : 0,
            
            "no_datasets_affils" : 0,
            "datasets_affils" : {
                Affil.ACADEMIC.name : 0,
                Affil.NON_PROFIT.name : 0,
                Affil.PROFIT.name : 0,
            },
            "num_datasets_affils" : 0,

            "All-Time SOTA_relative_score_inc" : 0,
            "Unique-Setting SOTA_relative_score_inc" : 0,
            "diffs" : {
                "All-Time SOTA" : [],
                "Unique-Setting SOTA" : []
            },
            "diffs_avg_over_paper" : {
                "All-Time SOTA" : [],
                "Unique-Setting SOTA" : []
            },

            "prior_best_affils" : {
                "Unique-Setting SOTA" : {
                    Affil.ACADEMIC.name : 0,
                    Affil.NON_PROFIT.name : 0,
                    Affil.PROFIT.name : 0
                },
                "Unique-Setting SOTA_num_pb" : 0,

                "All-Time SOTA" : {
                    Affil.ACADEMIC.name : 0,
                    Affil.NON_PROFIT.name : 0,
                    Affil.PROFIT.name : 0
                },
                "All-Time SOTA_num_pb" : 0,
                
            },

            "no_full_setting_affils" : 0,
            "full_setting_affils" : {
                Affil.ACADEMIC.name : 0,
                Affil.NON_PROFIT.name : 0,
                Affil.PROFIT.name : 0,
            },
            "num_full_setting_affils" : 0,
            "full_diffs" : [],

            "years" : {
                "datasets" : {},
                "ptms" : {},
                "prior_best" : {},
                "full_setting" : {},
            },

            "years_proc" : {
                "datasets" : {},
                "ptms" : {},
                "prior_best" : {},
                "full_setting" : {},
            },

            "num_metrics_per_paper" : 0,
            "percent_stat_sig" : 0,
            "percent_code" : 0

        }

        if self.affiliation is not None:
            self.get_paper_types()
            self.get_affils(name="basemodel")
            self.get_affils(name="datasets")
            self.get_average_relative_scores("All-Time SOTA")
            self.get_average_relative_scores("Unique-Setting SOTA")
            self.get_prior_best_affils("Unique-Setting SOTA", normalize=True)
            self.get_prior_best_affils("All-Time SOTA", normalize=True)
            self.get_full_setting_data(normalize=True)

        self.get_years()
        self.get_misc()

    def get_paper_types(self):
        for p in self.papers:
            self.paper_type_count(p["type"], p["number"])

    def paper_type_count(self, t, p_id):
        if type(t) is list:
            for val in t:
                self.paper_type_count(val, p_id)
            return
        if t not in self.data["paper_types"]:
            self.data["paper_types"][t] = 0
            self.data["ids_of_each_type"][t] = []
        self.data["paper_types"][t] += 1
        self.data["num_paper_types"] += 1
        self.data["ids_of_each_type"][t].append(p_id)


    def get_affils(self, name):
        for p in self.papers:

            if name not in p:
                self.data["no_" + name + "_affils"] += 1

            else:
                unique_names = [] 
                for b in p[name]:
                    if b["name"] in unique_names:
                        continue
                    unique_names.append(b["name"])

                    self.data[name + "_affils"][get_affil(b["affiliations"]).name] += 1
                    self.data["num_" + name + "_affils"] += 1


    def get_average_relative_scores(self, paper_type):

        num_papers = 0
        score_sum = 0

        for p in self.papers:
            diffs_total = 0
            diffs_sum = 0

            if paper_type not in self.data["ids_of_each_type"]:
                return

            if p["number"] not in self.data["ids_of_each_type"][paper_type]:
                continue

            for d in p["datasets"]:
                if "metrics" not in d:
                    continue

                for m in d["metrics"]:
                    diff = (m["new_score"] - m["old_score"])
                    if m["old_score"] != 0.00:
                        diff /= m["old_score"]
                    else:
                        diff = 1

                    if not m["higher_is_better"]:
                        diff = diff * -1
                    # if abs(diff) > 2: 
                    #     continue

                    self.data["diffs"][paper_type].append(diff)
                    diffs_sum += diff
                    diffs_total += 1

            # if diffs_sum/diffs_total < 0:
            #     continue

            score_sum += diffs_sum/diffs_total
            num_papers += 1
            self.data["diffs_avg_over_paper"][paper_type].append(diffs_sum/diffs_total)

        self.data[paper_type + "_relative_score_inc"] = \
        (score_sum/num_papers) *100


    def get_full_setting_data(self, normalize=True):

        total_sum = 0

        for p in self.papers:
            num_scores = 0
            paper_sum = 0

            if "Unique-Setting SOTA" not in self.data["ids_of_each_type"]:
                return

            if p["number"] not in \
                    self.data["ids_of_each_type"]["Unique-Setting SOTA"]:
                continue


            full_flag = 0

            for d in p["datasets"]:
                if "metrics" not in d:
                    continue

                for m in d["metrics"]:
                    if "full_setting" not in m:
                        self.data["no_full_setting_affils"] += 1
                        continue

                    if not (isinstance(m["full_setting"], list)):
                        self.data["full_setting_affils"]\
                                 [get_affil(m["full_setting"]\
                                    ["affiliations"]).name] += 1
                    else:
                        for p in m["full_setting"]:
                            self.data["full_setting_affils"]\
                                 [get_affil(p["affiliations"]).name]\
                                 += 1

                    full_flag = 1

                    diff = (m["new_score"] - m["full_score"])
                    if m["full_score"] != 0.00:
                        diff /= m["full_score"]
                    if not m["higher_is_better"]:
                        diff = diff * -1
                    # if abs(diff) >= 10:
                    #     continue
                    paper_sum += diff
                    num_scores += 1
                    self.data["full_diffs"].append(diff)

            self.data["num_full_setting_affils"] += num_scores


    def get_prior_best_affils(self, paper_type, normalize):


        if paper_type not in self.data["ids_of_each_type"]:
            return

        for p in self.papers:

            if p["number"] not in self.data["ids_of_each_type"][paper_type]:
                continue

            for d in p["datasets"]:
                if "metrics" not in d:
                    continue

                num_pb = 0
                for m in d["metrics"]:
                    if not (isinstance(m["prior_best"], list)):
                        self.data["prior_best_affils"][paper_type]\
                                 [get_affil(m["prior_best"]\
                                    ["affiliations"]).name] += 1
                        num_pb += 1 
                    else:
                        for p in m["prior_best"]:
                            self.data["prior_best_affils"][paper_type]\
                                 [get_affil(p["affiliations"]).name]\
                                 += 1
                            num_pb += 1 


                self.data["prior_best_affils"][paper_type + "_num_pb"] += \
                        num_pb


    def get_years(self):

        for p in self.papers:

            if "datasets" not in p:
                continue

            dataset_names = []

            for d in p["datasets"]:

                # ignore repeat datasets
                if d["name"] not in dataset_names:
                    if d["year"] not in self.data["years"]["datasets"]:
                        self.data["years"]["datasets"][d["year"]] = \
                                [get_affil(d["affiliations"])]
                    else:
                        self.data["years"]["datasets"][d["year"]].append(\
                            get_affil(d["affiliations"]))

                    dataset_names.append(d["name"])

                if "metrics" not in d:
                    continue

                for m in d["metrics"]:

                    if "prior_best" not in m:
                        continue
                    
                    pb = m["prior_best"]
                    if type(pb) is list:
                        for pb_element in pb:

                            if pb_element["year"] not in self.data["years"]["prior_best"]:
                                self.data["years"]["prior_best"][pb_element["year"]] = \
                                        [get_affil(pb_element["affiliations"])]
                            else:
                                self.data["years"]["prior_best"][pb_element["year"]].append(\
                                    get_affil(pb_element["affiliations"]))
                    else:
                        if pb["year"] not in self.data["years"]["prior_best"]:
                            self.data["years"]["prior_best"][pb["year"]] = \
                                    [get_affil(pb["affiliations"])]
                        else:
                            self.data["years"]["prior_best"][pb["year"]].append(\
                                get_affil(pb["affiliations"]))

                for m in d["metrics"]:

                    if "full_setting" not in m:
                        continue
                    
                    fs = m["full_setting"]
                    if type(fs) is list:
                        for fs_element in fs:
                            if fs_element["year"] not in self.data["years"]["full_setting"]:
                                self.data["years"]["full_setting"][fs_element["year"]] = \
                                        [get_affil(fs_element["affiliations"])]
                            else:
                                self.data["years"]["full_setting"][fs_element["year"]].append(\
                                    get_affil(fs_element["affiliations"]))

                    else:
                        if fs["year"] not in self.data["years"]["full_setting"]:
                            self.data["years"]["full_setting"][fs["year"]] = \
                                    [get_affil(fs["affiliations"])]
                        else:
                            self.data["years"]["full_setting"][fs["year"]].append(\
                                get_affil(fs["affiliations"]))


            if "basemodel" not in p:
                continue

            for b in p["basemodel"]:
                if b["year"] not in self.data["years"]["ptms"]:
                    self.data["years"]["ptms"][b["year"]] = \
                            [get_affil(b["affiliations"])]
                else:
                    self.data["years"]["ptms"][b["year"]].append(\
                        get_affil(b["affiliations"]))

        if None in self.data["years"]["ptms"]:
            del self.data["years"]["ptms"][None]
        if None in self.data["years"]["datasets"]:
            del self.data["years"]["datasets"][None]
        if None in self.data["years"]["prior_best"]:
            del self.data["years"]["prior_best"][None]
        if None in self.data["years"]["full_setting"]:
            del self.data["years"]["full_setting"][None]

    
        for item in self.data["years"]:
            for year, val_array in self.data["years"][item].items():

                i = num_ind = 0
                for v in val_array:
                    if v == Affil.PROFIT:
                        num_ind += 1
                    i += 1
                self.data["years_proc"][item][int(year)] = float(num_ind/i)
                

    def get_misc(self):

        num_stat_sig = 0
        num_code = 0
        total_metrics = 0

        for p in self.papers:

            if p["code"]:
                num_code += 1

            if p["significance"]:
                num_stat_sig += 1

            if "datasets" not in p:
                continue
                
            for d in p["datasets"]:

                if "metrics" not in d:
                    continue

                total_metrics += len(d["metrics"])
            
        self.data["num_metrics_per_paper"] = \
                    float(total_metrics/self.num_papers)
        self.data["percent_stat_sig"] = float(num_stat_sig/self.num_papers)
        self.data["percent_code"] = float(num_code/self.num_papers)


class Affil(Enum):
    ACADEMIC = 0
    NON_PROFIT = 1
    PROFIT = 2


def get_affil(affils):

    affil = Affil.ACADEMIC

    for author in affils:
        
        if type(author) is list:
            affil = get_affil(author)
        
        if (author == "P"):
            affil = Affil.PROFIT
            break
        elif (affil != Affil.NON_PROFIT):
            if (author == "N"):
                affil = Affil.NON_PROFIT

    return affil


def plot_stacked_bars(group_list, normalize=True,
            outside_legend=True, dpi=200, legend_font_size='x-small',
            save=False, save_name=None, title=None):

    # sns.set_context("talk")

    labels = []
    bar_labels = []
    y_values = {}
    bottom = np.zeros(3)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=dpi, figsize=(12, 10))
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, right=0.80, hspace=0.32, wspace=0.32)

    #basemodels plot

    for g in group_list:
        labels.append(g.affiliation.name)
        bar_labels.append("n=" + str(g.data["num_" + "basemodel_affils"]))

        if normalize:
            for d in g.data["basemodel_affils"]:
                g.data["basemodel_affils"][d] = \
                        g.data["basemodel_affils"][d]/g.data["num_" + "basemodel_affils"]
    
        for d in g.data["basemodel_affils"]:
            if d not in y_values:
                y_values[d] = np.zeros(3)
            y_values[d][g.affiliation.value] = g.data["basemodel_affils"][d]*100


    for x, y in y_values.items():
        x = convert_enum_to_text(x)
        labels = [convert_enum_to_text(x) for x in labels]
        b = ax1.bar(labels, y, width=0.5, label=x, bottom=bottom)
        bottom += y

    ax1.tick_params(axis='y', which='major', labelsize=12)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    ax1.set_ylim(0, 110)
    ax1.bar_label(b,  fontsize=12, padding=4, labels=bar_labels)
    ax1.grid(True, axis="y")
    ax1.set_title("PTMs", fontsize=16, font="Courier New", pad=18)
    ax1.set_ylabel("Percent", fontsize=14, labelpad=10, font="Courier New")

    # datasets plot

    labels = []
    bar_labels = []
    y_values = {}
    bottom = np.zeros(3)

    for g in group_list:
        labels.append(g.affiliation.name)
        bar_labels.append("n=" + str(g.data["num_" + "datasets_affils"]))

        if normalize:
            for d in g.data["datasets_affils"]:
                g.data["datasets_affils"][d] = \
                        g.data["datasets_affils"][d]/g.data["num_" + "datasets_affils"]
    
        for d in g.data["datasets_affils"]:
            if d not in y_values:
                y_values[d] = np.zeros(3)
            y_values[d][g.affiliation.value] = g.data["datasets_affils"][d]*100


    for x, y in y_values.items():
        x = convert_enum_to_text(x)
        labels = [convert_enum_to_text(x) for x in labels]
        b = ax2.bar(labels, y, width=0.5, label=x, bottom=bottom)
        bottom += y

    ax2.tick_params(axis='y', which='major', labelsize=12)
    ax2.tick_params(axis='x', which='major', labelsize=12)
    ax2.set_ylim(0, 110)
    ax2.bar_label(b,  fontsize=12, padding=4, labels=bar_labels)
    ax2.grid(True, axis="y")
    ax2.set_title("Datasets", fontsize=16, font="Courier New", pad=18)
    ax2.set_ylabel("Percent", fontsize=14, labelpad=10, font="Courier New")

    y_values = {}
    labels = []
    bar_labels = []
    bottom = np.zeros(3)

    for g in group_list:
        if normalize:
            for d in g.data["prior_best_affils"]["Unique-Setting SOTA"]:
                g.data["prior_best_affils"]["Unique-Setting SOTA"][d] = \
                        g.data["prior_best_affils"]["Unique-Setting SOTA"][d]\
                        /g.data["prior_best_affils"]["Unique-Setting SOTA_num_pb"]

            if "All-Time SOTA" in g.data["paper_types"]:
                for d in g.data["prior_best_affils"]["All-Time SOTA"]:
                    g.data["prior_best_affils"]["All-Time SOTA"][d] = \
                            g.data["prior_best_affils"]["All-Time SOTA"][d]\
                            /g.data["prior_best_affils"]["All-Time SOTA_num_pb"]

        labels.append(g.affiliation.name)
        bar_labels.append("n=" +
                str(g.data["prior_best_affils"]["Unique-Setting SOTA_num_pb"] +\
                    g.data["prior_best_affils"]["All-Time SOTA_num_pb"]))
   
        for a in g.data["prior_best_affils"]["Unique-Setting SOTA"]:
            if a not in y_values:
                y_values[a] = np.zeros(3)
            if g.data["prior_best_affils"]["All-Time SOTA"][a] != 0:
                y_values[a][g.affiliation.value] = \
                        g.data["prior_best_affils"]["Unique-Setting SOTA"][a]*100/2
            else:
                y_values[a][g.affiliation.value] = \
                        g.data["prior_best_affils"]["Unique-Setting SOTA"][a]*100

        for a in g.data["prior_best_affils"]["All-Time SOTA"]:
            if a not in y_values:
                y_values[a] = np.zeros(3)
            y_values[a][g.affiliation.value] += \
                    g.data["prior_best_affils"]["All-Time SOTA"][a]*100/2


    for x, y in y_values.items():
        x = convert_enum_to_text(x)
        labels = [convert_enum_to_text(x) for x in labels]
        b = ax3.bar(labels, y, width=0.5, label=x, bottom=bottom)
        bottom += y

    ax3.tick_params(axis='y', which='major', labelsize=12)
    ax3.tick_params(axis='x', which='major', labelsize=12)
    ax3.set_ylim(0, 110)
    ax3.bar_label(b,  fontsize=12, padding=4, labels=bar_labels)
    ax3.grid(True, axis="y")
    ax3.set_title("Prior Bests", fontsize=16, font="Courier New", pad=18)
    ax3.set_ylabel("Percent", fontsize=14, labelpad=10, font="Courier New")

    y_values = {}
    labels = []
    bar_labels = []
    bottom = np.zeros(3)

    for g in group_list:
        labels.append(g.affiliation.name)
        bar_labels.append("n=" + str(g.data["num_" + "full_setting_affils"]))

        if normalize:
            for d in g.data["full_setting_affils"]:
                g.data["full_setting_affils"][d] = \
                        g.data["full_setting_affils"][d]/g.data["num_" + "full_setting_affils"]

        for d in g.data["full_setting_affils"]:
            if d not in y_values:
                y_values[d] = np.zeros(3)
            y_values[d][g.affiliation.value] = \
                    g.data["full_setting_affils"][d]*100


    for x, y in y_values.items():
        x = convert_enum_to_text(x)
        labels = [convert_enum_to_text(x) for x in labels]
        b = ax4.bar(labels, y, width=0.5, label=x, bottom=bottom)
        bottom += y

    ax4.tick_params(axis='y', which='major', labelsize=12)
    ax4.tick_params(axis='x', which='major', labelsize=12)
    ax4.set_ylim(0, 110)
    ax4.bar_label(b,  fontsize=12, padding=4, labels=bar_labels)
    ax4.grid(True, axis="y")
    ax4.set_title("Full Settings", fontsize=16, font="Courier New", pad=18)
    ax4.set_ylabel("Percent", fontsize=14, labelpad=10, font="Courier New")

    handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]

    handles = handles[:3]
    labels = labels[:3]


    if outside_legend:
        fig.legend(handles, labels, loc='center right', #ncol=4,
                bbox_to_anchor = (0, 0, 0.97, 0.96), 
                bbox_transform = plt.gcf().transFigure,
                fontsize=12)
    else:
        fig.legend(fontsize=legend_font_size)

    if title is not None:
        plt.title(title)

    if save:
        if save_name is not None:
            save_path = os.path.join("outputs", save_name)
            plt.savefig(save_path, bbox_inches='tight')
        else:
            print("Error: pass file name via 'save_name' argument if "\
                  "saving")
    else:
        plt.show()

    
def plot_box_single(groups, log_scale=False, save=False, save_name=None,
        remove_negatives=False, remove_outliers=False,
        thresh_scale=2, t_test=True):

    fig, ax = plt.subplots(dpi=200, figsize=(6, 5))

    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12, rotation=45)

    if log_scale:
        ax.set_xscale('log')

    all_data = []
    labels = []

    for g in groups:
        labels.append(g.affiliation.name)
        filtered_data = []
        if len(g.data["diffs_avg_over_paper"]["All-Time SOTA"]) > 0:
            for diff in g.data["diffs_avg_over_paper"]["All-Time SOTA"]:
                diff *= 100

                if remove_negatives:
                    if diff < 0:
                        continue
                filtered_data.append(diff)

        for diff in g.data["diffs_avg_over_paper"]["Unique-Setting SOTA"]:
            diff *= 100

            if remove_negatives:
                if diff < 0:
                    continue
            filtered_data.append(diff)

        all_data.append(filtered_data)

    if remove_outliers:
        std_devs = []
        means = []
        # for affil in all_data:
        #     std_devs.append(statistics.stdev(affil))
        #     means.append(statistics.mean(affil))

        std_devs = [statistics.stdev(all_data[0] + all_data[1] + all_data[2])]
        means = [statistics.mean(all_data[0] + all_data[1] + all_data[2])]
        std_devs = std_devs*3
        thresholds = [x*thresh_scale for x in std_devs]
        means = means*3

        for i, affil in enumerate(all_data):
            lb = means[i] - thresholds[i] 
            ub = means[i] + thresholds[i] 
            for j, diff in enumerate(affil):
                if diff < lb or diff > ub:
                    affil.pop(j)

    labels = [convert_enum_to_text(x) for x in labels]

    bp_dict = ax.boxplot(all_data, widths=0.3, showmeans=True, vert=False)
    ax.set_yticks([y + 1 for y in range(len(all_data))], labels=labels)
    ax.grid(True, axis="x")
    # ax.set_title("All-Time", fontsize=16, font="Courier New", pad=18)
    ax.set_xlabel("Absolute Score Increase", fontsize=14, labelpad=10, font="Courier New")

    for point in bp_dict['means']:
        x, y = point.get_xydata()[0]
        ax.annotate('%.1f' % x, (x, y), xytext=[x, y-0.3], 
                horizontalalignment='center', fontsize=12, font="Courier New")

    if save:
        if save_name is not None:
            save_path = os.path.join("outputs", save_name)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            print("Error: pass file name via 'save_name' argument if "\
                  "saving")
    else:
        plt.show()

    if t_test:
        print("Combined T-Test Results:")
        print(ttest_ind(all_data[0], all_data[2]))


def plot_box(groups, log_scale=False, save=False, save_name=None,
        remove_negatives=False, remove_outliers=False,
        thresh_scale=2, t_test=True):


    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=200, figsize=(6, 9),
            gridspec_kw={'height_ratios': [2, 3]})

    plt.subplots_adjust(hspace=0.5)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    ax2.tick_params(axis='x', which='major', labelsize=12)
    ax1.tick_params(axis='y', which='major', labelsize=12, rotation=45)
    ax2.tick_params(axis='y', which='major', labelsize=12, rotation=45)

    if log_scale:
        ax1.set_xscale('log')
        ax2.set_xscale('log')

    all_data = []
    labels = []

    for g in groups:
        if len(g.data["diffs_avg_over_paper"]["All-Time SOTA"]) > 0:
            labels.append(g.affiliation.name)
            filtered_data = []

            for diff in g.data["diffs_avg_over_paper"]["All-Time SOTA"]:
                diff *= 100

                if remove_negatives:
                    if diff < 0:
                        continue
                filtered_data.append(diff)

            all_data.append(filtered_data)

    if remove_outliers:
        std_devs = []
        means = []
        # for affil in all_data:
        #     std_devs.append(statistics.stdev(affil))
        #     means.append(statistics.mean(affil))

        std_devs = [statistics.stdev(all_data[0] + all_data[1])]
        means = [statistics.mean(all_data[0] + all_data[1])]
        std_devs = std_devs*2
        thresholds = [x*thresh_scale for x in std_devs]
        means = means*2

        for i, affil in enumerate(all_data):
            lb = means[i] - thresholds[i] 
            ub = means[i] + thresholds[i] 
            for j, diff in enumerate(affil):
                if diff < lb or diff > ub:
                    affil.pop(j)

    labels = [convert_enum_to_text(x) for x in labels]

    bp_dict = ax1.boxplot(all_data, widths=0.3, showmeans=True, vert=False)
    ax1.set_yticks([y + 1 for y in range(len(all_data))], labels=labels)
    # ax1.set_xticks([])
    ax1.grid(True, axis="x")
    ax1.set_title("All-Time", fontsize=16, font="Courier New", pad=18)
    ax1.set_xlabel("Percent", fontsize=14, labelpad=10, font="Courier New")

    for point in bp_dict['means']:
        x, y = point.get_xydata()[0]
        ax1.annotate('%.1f' % x, (x, y), xytext=[x, y-0.3], 
                horizontalalignment='center', fontsize=12, font="Courier New")


    unique_data = []
    labels = []

    for g in groups:
        if len(g.data["diffs_avg_over_paper"]["Unique-Setting SOTA"]) > 0:
            labels.append(g.affiliation.name)

            filtered_data = []
            for diff in g.data["diffs_avg_over_paper"]["Unique-Setting SOTA"]:
                diff *= 100

                if remove_negatives:
                    if diff < 0:
                        continue
                filtered_data.append(diff)

            unique_data.append(filtered_data)

    if remove_outliers:
        std_devs = []
        means = []
        # for affil in unique_data:
        #     std_devs.append(statistics.stdev(affil))
        #     means.append(statistics.mean(affil))

        std_devs = [statistics.stdev(unique_data[0] + unique_data[1] + \
                unique_data[2])]
        means = [statistics.mean(unique_data[0] + unique_data[1] + \
            unique_data[2])]
        std_devs = std_devs*3
        thresholds = [x*thresh_scale for x in std_devs]
        means = means*3
        
        for i, affil in enumerate(unique_data):
            lb = means[i] - thresholds[i] 
            ub = means[i] + thresholds[i] 
            for j, diff in enumerate(affil):
                if diff < lb or diff > ub:
                    affil.pop(j)

    labels = [convert_enum_to_text(x) for x in labels]

    bp_dict = ax2.boxplot(unique_data, widths=0.3, showmeans=True, vert=False)

    ax2.set_yticks([y + 1 for y in range(len(unique_data))], labels=labels)
    ax2.grid(True, axis="x")
    ax2.set_title("Unique-Setting", fontsize=16, font="Courier New", pad=18)
    ax2.set_xlabel("Percent", labelpad=10, font="Courier New", fontsize=14)
    for point in bp_dict['means']:
        x, y = point.get_xydata()[0]
        ax2.annotate('%.1f' % x, (x, y), xytext=[x, y-0.3], 
                horizontalalignment='center', font="Courier New", fontsize=12)


    if save:
        if save_name is not None:
            save_path = os.path.join("outputs", save_name)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            print("Error: pass file name via 'save_name' argument if "\
                  "saving")
    else:
        plt.show()


    if t_test:
        print("All-Time T-Test Results:")
        print(ttest_ind(all_data[0], all_data[1]))
        print("Unique Setting T-Test Results:")
        print(ttest_ind(unique_data[0], unique_data[2]))
        
        
def plot_years(papers, smooth=None, save=False, save_name=None):

    fig, ax = plt.subplots(dpi=200, figsize=(10, 7))
    #fig.subplots_adjust(top=0.85)

    x_list = []
    y_list = []
    labels = []
    for item in papers.data["years_proc"]:
        x_list.append(list(dict(sorted(\
                papers.data["years_proc"][item].items())).keys()))
        y_list.append(list(dict(sorted(\
                papers.data["years_proc"][item].items())).values())) 
        labels.append(item)

    labels = convert_labels_to_text(labels)

    scaled_y_list = []
    for y in y_list:
        scaled_y_list.append([y_element*100 for y_element in y])

    y_list = scaled_y_list

    expected_x = []
    expected_y = []

    expected_file = os.path.join("data", "expected_industry.csv")
    with open(expected_file, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                continue
            expected_x.append(float(row[0]))
            expected_y.append(float(row[1])*100)

    x_list.append(expected_x)
    y_list.append(expected_y)
    labels.append("Expected")



    if smooth == 'gaussian':
        for x, y, label in zip(x_list, y_list, labels):
            ysmoothed = gaussian_filter1d(y, sigma=2.5)

            if label == "Expected":
                ax.plot(x, ysmoothed, linestyle="dashed", color="grey", linewidth=2,
                        label=label)
            else:
                ax.plot(x, ysmoothed, label=label, linewidth=2)

    elif smooth == 'spline':
        for x, y, label in zip(x_list, y_list, labels):
            xnew = np.linspace(min(x), max(x), 300)
            spl = make_interp_spline(x, y, k=3)
            ynew = spl(xnew)
            ax.plot(xnew, ynew, label=label)

    elif smooth == 'gaussian+spline':
        for x, y, label in zip(x_list, y_list, labels):
            ysmoothed = gaussian_filter1d(y, sigma=3)
            xnew = np.linspace(min(x), max(x), 300)
            spl = make_interp_spline(x, ysmoothed, k=3)
            ynew = spl(xnew)
            ax.plot(xnew, ynew, label=label)

    else:
        for x, y, label in zip(x_list, y_list, labels):
            ax.plot(x, y, label=label)
        
    ax.tick_params(axis='x', which='major', rotation=45)
    ax.set_ylabel("Percent", labelpad=10, weight='bold')
    ax.grid(True, axis="y")
    ax.legend(loc='upper left', fontsize=16, ncol=2, facecolor="whitesmoke")
            # bbox_to_anchor = (0, 0, 1.02, 0.99), 
            # bbox_transform = plt.gcf().transFigure)

    if save:
        if save_name is not None:
            save_path = os.path.join("outputs", save_name)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            print("Error: pass file name via 'save_name' argument if "\
                  "saving")
    else:
        plt.show()
    

def convert_enum_to_text(enum_string):
    if enum_string == "NON_PROFIT":
        return "Non-Profit"
    if enum_string == "PROFIT":
        return "Industry"
    if enum_string == "ACADEMIC":
        return "Academia"
    else:
        return enum_string

def convert_enum_to_text_abrv(enum_string):
    if enum_string == "NON_PROFIT":
        return "NP"
    if enum_string == "PROFIT":
        return "Ind"
    if enum_string == "ACADEMIC":
        return "Acd"
    else:
        return enum_string

def convert_labels_to_text(labels):
    new = []

    for l in labels:
        if l == "full_setting":
            new.append("Full Setting")
        elif l == "prior_best":
            new.append("Prior Best")
        elif l == "ptms":
            new.append("PTMs")
        elif l == "datasets":
            new.append("Datasets")

    return new

if __name__ == "__main__":
    main()
