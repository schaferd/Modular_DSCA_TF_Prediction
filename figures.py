import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
import numpy as np
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
plt.rcParams['agg.path.chunksize'] = 10000


def create_moa_figs(moa_train_losses,moa_test_losses,moa_violation_count_train,moa_violation_count_test,save_path,fold=0,cycle=0):
    fig = plt.figure(5)
    plt.plot(np.array(moa_train_losses),label="train loss")
    plt.plot(np.array(moa_test_losses),label="test loss")
    plt.title("MOA Loss fold"+str(fold))
    plt.xlabel('epochs')
    plt.ylabel('moa loss')
    plt.savefig(save_path+'/moa_loss_cycle'+str(cycle)+'_fold'+str(fold)+".png")
    plt.clf()

    fig = plt.figure(5)
    plt.plot(np.array(moa_violation_count_train),label="train violations")
    plt.plot(np.array(moa_violation_count_test),label="test violations")
    plt.title("MOA Violations fold"+str(fold))
    plt.xlabel('epochs')
    plt.ylabel('num violations')
    plt.legend()
    plt.savefig(save_path+'/moa_violations_cycle'+str(cycle)+'_fold'+str(fold)+".png")
    plt.clf()

def flatten_list(l):
        """
        Turns a 2D list l into a 1D list
        """
        new_list = []
        counter = 0
        for l_ in l:
                for item in l_:
                    new_list.append(item)
                    counter += 1
        return new_list

def plot_input_vs_output(ae_input,ae_output,corr,is_test,save_path,fold=0,cycle=0):
    """
    Saves a scatter plot to demostrate correlation between input vs. output 
    """
    new_input = np.asarray(flatten_list(ae_input)).flatten()
    new_output = np.asarray(flatten_list(ae_output)).flatten()
    ax_min = min(min(new_input),min(new_output))
    ax_max = max(max(new_input),max(new_output))
    plt.hist2d(x=new_input,y=new_output,bins=200,norm=mpl.colors.LogNorm())
    plt.ylabel('Output',size=12)
    plt.xlabel('Input',size=12)
    plt.xlim(ax_min,ax_max)
    plt.ylim(ax_min,ax_max)
    #plt.axis('square')
    if is_test:
        plt.title('AE Input vs. Output: Test Correlation:'+str(corr))
        plt.savefig(save_path+'/test_y_vsyhat_scatter_cycle'+str(cycle)+'_fold'+str(fold)+'_corr'+str(corr)+'.png')
        with open(save_path+'/test_input_cycle'+str(cycle)+'_fold'+str(fold)+'_corr'+str(corr)+'.pkl','wb') as f:
            pkl.dump(new_input,f)
        with open(save_path+'/test_output_cycle'+str(cycle)+'_fold'+str(fold)+'_corr'+str(corr)+'.pkl','wb') as f:
            pkl.dump(new_output,f)
    else:
        #plt.title('AE Input vs. Output: Train Correlation'+str(corr))
        #plt.savefig(save_path+'/train_y_vsyhat_scatter_cycle'+str(cycle)+'_fold'+str(fold)+'_corr'+str(corr)+'.png')
        with open(save_path+'/train_input_cycle'+str(cycle)+'_fold'+str(fold)+'_corr'+str(corr)+'.pkl','wb') as f:
            pkl.dump(new_input,f)
        with open(save_path+'/train_output_cycle'+str(cycle)+'_fold'+str(fold)+'_corr'+str(corr)+'.pkl','wb') as f:
            pkl.dump(new_output,f)
    plt.clf()

def create_test_vs_train_plot(training_losses,test_losses,save_path,fold=0,cycle=0):
    """
    Saves RMSE vs. Epochs line plot
    """
    plt.clf()
    fig = plt.figure(4)
    training_losses = [tensor for tensor in training_losses[fold]]# if type(tensor) == torch.Tensor]
    test_losses = [tensor for tensor in test_losses[fold]]
    tr_rmse = training_losses[-1]
    test_rmse = test_losses[-1]

    plt.plot(np.array(training_losses),label="Train")
    plt.plot(np.array(test_losses),label="Test")
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title("Train RMSE "+str(round(tr_rmse,2))+" Test RMSE "+str(round(test_rmse,2)))
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend()
    plt.savefig(save_path+'/rmse_cycle'+str(cycle)+'_fold'+str(fold)+'.png')
    plt.clf()

def create_corr_hist(corr_list,corr,save_path,is_test,fold=0,cycle=0):
        """
        Saves box plot that shows distribution of correlations across all genes
        """
        plt.clf()
        plt.hist(corr_list,bins=5)
        if is_test:
            plt.title('Test Correlation Distribution')
            plt.savefig(save_path+'/test_corr_hist_cycle'+str(cycle)+'_fold'+str(fold)+'.png')
        else:
            plt.title('Train Correlation Distribution')
            plt.savefig(save_path+'/train_corr_hist_cycle'+str(cycle)+'_fold'+str(fold)+'.png')
        plt.clf()

def TF_ko_heatmap(df,save_path,metric_type,fold=0,cycle=0):
    plt.clf()
    #df should be TF by KO_TF and heat metric should be rank or tf activity val
    ax = sns.heatmap(df)
    plt.ylabel('KO\'ed TF')
    plt.xlabel('TF')
    plt.title(metric_type)
    plt.savefig(save_path+'/ko_heatmap_'+metric_type+'_cycle'+str(cycle)+'_fold'+str(fold)+'.png')
    plt.clf()
    #df should be TF by KO_TF and heat metric should be rank or tf activity val
    ax = sns.clustermap(df)
    plt.ylabel('KO\'ed TF')
    plt.xlabel('TF')
    plt.title(metric_type)
    plt.savefig(save_path+'/ko_clustermap_'+metric_type+'_cycle'+str(cycle)+'_fold'+str(fold)+'.png')


def consistency_scatter(reconstructions):
    plt.clf()
    reconstructions = np.array(reconstructions)
    std_r = np.std(reconstructions,axis=0)
    mean_r = np.mean(reconstructions,axis=0)
    x_axis = list(range(len(reconstructions[0])))

    plt.scatter(x_axis,mean_r)
    plt.errorbar(x_axis,mean_r,y_err= std_r)
    plt.savefig(save_path+'/consistency_scatter.png')

def CV_boxplot(best_cv_path, fc_cv_path):
    with open(best_cv_path) as f:
        best_cv = pkl.load(f)
    with open(fc_cv_path) as f:
        fc_cv = pkl.load(f)
    plt.clf()
    data = np.array([best_cv, fc_cv])
    labels = ['Sparse','Fully Connected']
    fig,ax = plt.subplots()
    ax.violinplot(data)
    ax.set_xticks(np.arange(1,len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Coefficient of Variation')
    ax.set_xlabel('Model Type')

    ax.set_title('Consistency CV:'+str(np.mean(CV))+' null CV:'+str(np.mean(null_CV)))
    plt.savefig(self.savedir+'/CV_boxplot.png')

def AUC_boxplot(save_path):
    self.AUC_paths = [save_dir+'/'+f for f in os.listdir(save_dir) if os.path.isfile(save_dir+'/'+f) and '.pth' in f]

