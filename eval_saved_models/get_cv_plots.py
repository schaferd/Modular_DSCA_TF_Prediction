import pandas as pd
import matplotlib.pyplot as plt
import os 
import sys
import pickle as pkl
import numpy as np
from get_pert_roc_curves import getPertROC
from get_roc_curve import getROC


class CVPlot:

    def __init__(self,out_paths):
        self.out_paths = out_paths
        #self.run_pert_tests()
        self.run_ko_tests(ranked=True)




    def get_data(self,out_dir,activities_search_term='/diff_activities.csv', ko_tf_search_term='/ko_tf_index.pkl'):
        raw_activities = []
        index_to_ko_tfs = []

        activity_paths = [out_dir+'/'+f for f in os.listdir(out_dir) if "fold" in f and "cycle" in f and os.path.isfile(out_dir+f) == False]
        ko_activity_paths = []
        for i, f in enumerate(activity_paths):
            paths = os.listdir(f)
            for p in paths:
                if "ko_activities" in p:
                    ko_activity_paths.append(f+'/'+p+'/')

        for i in ko_activity_paths:
            raw_activities.append(pd.read_csv(i+activities_search_term,index_col=0))
            with open(i+ko_tf_search_term,'rb') as f:
                index_to_ko_tfs.append(pkl.load(f))

        return raw_activities, index_to_ko_tfs


    def getCVs(self, activities):
        activities = np.array([a.copy().to_numpy() for a in activities])
        mean = activities.mean(axis=0)+0.00001
        std = activities.std(axis=0)
        cv = abs(std/mean)
        sample_avg_cv = cv.mean(axis=1)
        return sample_avg_cv, cv

    def generate_cutoff_buckets(self,index_list,sample_avg_cv,num_buckets=10):
        print(sample_avg_cv.max(), sample_avg_cv.max()/num_buckets)
        bucket_labels = np.arange(0,sample_avg_cv.max(), sample_avg_cv.max()/num_buckets)
        buckets = {}
        
        for bucket in bucket_labels:
            buckets[bucket] = []

        for i,cv in enumerate(sample_avg_cv):
            for bucket in buckets.keys():
                if cv < bucket:
                    buckets[bucket].append(index_list[i])

        return buckets

    def rank_matrix(self, matrix):
        ranked_matrix = matrix.rank(axis = 1,method='min',na_option='keep',ascending=True)
        ranked_matrix = ranked_matrix.reset_index(drop=True)
        scaled_rank_matrix = (ranked_matrix.T/ranked_matrix.max(axis=1)).T
        scaled_rank_matrix.index = matrix.index
        return scaled_rank_matrix

    def run_pert_tests(self, ranked=False):
        activities = []
        index_to_ko_tfs = []

        for d in self.out_paths:
            a, i_to_kotfs = self.get_data(d)
            activities.extend(a)
            index_to_ko_tfs.extend(i_to_kotfs)

        values = activities
        if ranked:
            values = [self.rank_matrix(a) for a in activities]

        index_to_ko_df = [pd.DataFrame(ko_tfs,index=['KO_TF']).T for ko_tfs in index_to_ko_tfs]

        sample_avg_cv, cv = self.getCVs(values)
        buckets = self.generate_cutoff_buckets(index_to_ko_df.index,sample_avg_cv,num_buckets=20)
        print(buckets)

        aucs = []
        bucket_labels = []
        for bucket in buckets.keys():
            indices = buckets[bucket]
            if len(indices) > 0:
                avg_activities = None
                ko_df = None
                for i,a in enumerate(activities):
                    if avg_activities is None:
                        avg_activities = a.iloc[indices,:]
                    else:
                        avg_activities = avg_activities.add(a.iloc[indices,:])

                    ko_df = index_to_ko_df[i].iloc[indices,:]
                avg_activities = avg_activities / len(activities) 

                ko_df = ko_df.T.to_dict(orient='list')
                ko_df = {key:ko_df[key][0] for key in ko_df.keys()}

                pert = getPertROC(avg_activities,ko_df,'pert_cv.png')
                aucs.append(pert.auc)
                bucket_labels.append(bucket)
            else:
                print("skipping bucket")


        plt.clf()
        plt.plot(bucket_labels,aucs)
        plt.savefig('outputs/pert_cv_plot.png')



    def run_ko_tests(self, ranked=False):
        activities = []
        index_to_ko_tfs = []

        for d in self.out_paths:
            a, i_to_kotfs = self.get_data(d,activities_search_term='/knockout_diff_activities.csv', ko_tf_search_term='/knocktf_sample_to_tf.pkl')
            activities.extend(a)
            index_to_ko_tfs.extend(i_to_kotfs)

        print(activities[0])
        values = activities
        if ranked:
            values = [self.rank_matrix(a) for a in activities]
        

        samples = activities[0].index.tolist()
        print(activities[0].to_string())
        raise ValueError()
        for a in activities:
            print(a.index.tolist() == samples)


        sample_avg_cv, cv = self.getCVs(values)
        buckets = self.generate_cutoff_buckets(activities[0].index.tolist(),sample_avg_cv,num_buckets=20)
        print(buckets)

        aucs = []
        bucket_labels = []
        for bucket in buckets.keys():
            indices = buckets[bucket]
            if len(indices) > 0:
                avg_activities = None
                ko_df = None
                for i,a in enumerate(activities):
                    print(a.index.tolist())
                    print(indices)
                    print(a.loc['DataSet_01_119',:])
                    raise ValueError()
                    if avg_activities is None:
                        avg_activities = a.loc[indices,:]
                    else:
                        avg_activities = avg_activities.add(a.loc[indices,:])
                    print(avg_activities)
                    raise ValueError()

                    ko_df = index_to_ko_tfs[i].loc[indices,:]
                print(avg_activities)
                avg_activities = avg_activities / len(activities) 

                print(avg_activities)
                raise ValueError()

                pert = getROC(avg_activities,list(ko_df['TF']),'pert_cv.png')
                aucs.append(pert.auc)
                bucket_labels.append(bucket)
            else:
                print("skipping bucket")


        plt.clf()
        plt.plot(bucket_labels,aucs)
        plt.savefig('outputs/ko_cv_plot.png')





        

if __name__ == '__main__':
    obj = CVPlot(['/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/'])





