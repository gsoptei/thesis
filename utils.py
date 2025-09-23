# libraries for geospatial analysis, data cleaning and data analysis
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.lines as mlines
import io
import seaborn as sns
from datetime import datetime
from osgeo import ogr, osr, gdal
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn import svm
from settings import *
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import OneClassSVM
from sklearn.inspection import DecisionBoundaryDisplay
from seaborn import pairplot
from settings import *

# Define functions

# Auxiliary class to silence warnings
class SuppressSettingWithCopyWarning:
    def __init__(self, arg):
        self._arg = arg
        
    def __enter__(self):
        pd.options.mode.chained_assignment = None
            
    def __call__(self, *args):
        pd.options.mode.chained_assignment = None
        return self._arg(*args)

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = 'warn'
        

def report_best_scores(results, n_top=3):
    param_list = []
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        print("*" * 15)
        print(f'These are the candidates: {candidates}')
        print("*" * 15)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            print(f"-------------------------------------- {results['mean_test_score'][candidate]}")
            hyperparams = results['params'][candidate]
            param_list.append(hyperparams)

    return param_list

# Extract raster values for points
def extract_raster_values_at_points(year, rasters, points_gdf, phen_vars, season="s1", main_dir=MAIN_DIR, working_dir=WORKING_DIR):    
    
    # variables
    dataset = rasters.copy()
    rasters_year = [raster for raster in dataset if year in raster and season in raster]
    new_gdf = points_gdf.copy()
    my_vars = phen_vars.copy()
    
    # loop through raster dataset and extract the values at dead tree points
    for raster in rasters_year:
        for var in my_vars:
            if var in raster:
                values = []
                phen_raster = rasterio.open(main_dir+working_dir+raster)
                my_vars.remove(var)
                for index, point in new_gdf.iterrows():
                    x, y = point.geometry.x, point.geometry.y
                    row, col = phen_raster.index(x, y)
                    value = phen_raster.read(1, window=((row, row+1), (col, col+1)))
                    values.append(value[0, 0])
                new_gdf[var] = values

    # subset data with phenology data            
    # new_gdf = new_gdf[phen_vars]
    return new_gdf

# Drop rows where rasters contained None values 
def drop_nulls(dataframe):
    
    counter = 0
    # drop empty values and subset the dataframe
    new_gdf = dataframe[dataframe['SPROD'] != 65535]
    # check for 0 values
    count_nan = new_gdf.isnull().sum()
    for i in count_nan:
        if i == 0:
            counter += 1
    if counter == len(count_nan):
        return new_gdf
    else:
        # print('Hmm....you have 0s dar!')
        return new_gdf

# Format dates so that they represent days with regards to the start of the year in question
@SuppressSettingWithCopyWarning
def format_dates(dataframe, year, dates=['EOSD', 'MAXD', 'SOSD']):

    start = int(year[2:])
    the_data = dataframe.copy()
    for i in dates:
        for j in the_data.index:
            try:
                if str(the_data[i][j])[:2] == str(int(start) - 1):
                    subtract = int(str(the_data[i][j])[2:])
                    the_data[i][j] = subtract - 365
                else:
                    the_data[i][j] = int(str(the_data[i][j])[2:])
            except:
                print(f'Index {j} is not in the dataframe')
    return the_data

def add_dead(dataframe, dead=True):
    the_data = dataframe.copy()
    if dead:
        the_data['DEAD'] = 1
    else: 
        the_data['DEAD'] = -1 
    return the_data

def subset_data(dataframe, phen_vars=PHEN_VARS):
    
    phen = phen_vars.copy()
    # add diffrence of TRPOD and SPROD to variables
    phen.append('PROD_DIFF')
    # get feature data
    X_train = dataframe.copy()
    X_train['PROD_DIFF'] = X_train['TPROD'] - X_train['SPROD']
    X = X_train[phen].to_numpy()
    
    # get dead tree data
    y = X_train['DEAD'].to_numpy()

    return X, y, X_train

def search_hyperparameters(parameters, X, y, X_living, y_living, results_path):    

    svm_model = OneClassSVM()
    grid_search = GridSearchCV(estimator=svm_model,
                                       param_grid=parameters,
                                       # scoring='accuracy',
                                       # scoring='recall',
                                       # scoring='top_k_accuracy',
                                       scoring='f1',
                                       cv = 10,
                                       n_jobs= -1,
                                       # refit=False
                                       refit=True)
    
    print('SEARCHING HYPERPARAMETERS\n')
    grid_search.fit(X, y)
    hyperparams = report_best_scores(grid_search.cv_results_, 5)
    df = pd.DataFrame(grid_search.cv_results_)
    df.to_excel(Path(results_path, "%s_GridSearchResults_DEAD.xlsx" % (datetime.now().strftime("%Y%m%d-%H%M%S"))))
    print('DEAD TREE MODELS SAVED')
    print(f'These are the best hyperparameters: {hyperparams}')

    grid_search_live = GridSearchCV(estimator=svm_model,
                                       param_grid=parameters,
                                       scoring='accuracy',
                                       cv = 10,
                                       n_jobs= -1,
                                       refit=True)
    
    print('SEARCHING HYPERPARAMETERS FOR LIVING TREES\n')
    grid_search_live.fit(X_living, y_living)
    living_params = report_best_scores(grid_search_live.cv_results_, 5)
    df_living = pd.DataFrame(grid_search_live.cv_results_)
    # df_living._append(grid_search_live.best_estimator_.feature_names_in_)
    df_living.to_excel(Path(results_path, "%s_GridSearchResults_LIVING.xlsx" % (datetime.now().strftime("%Y%m%d-%H%M%S"))))
    print('LIVING TREE MODELS SAVED')
    
    return hyperparams

def plot_decision_boundaries(X_train, X_test, results_path, year, features, params, phen_vars, dataframe, to_drop=TO_DROP, living=False):
    # create sub datasets    
    X_train = X_train[phen_vars]
    X_train['PROD_DIFF'] = X_train['TPROD'] - X_train['SPROD']
    X_test = X_test[phen_vars]
    X_test['PROD_DIFF'] = X_test['TPROD'] - X_test['SPROD']

    # modify the copy of the original dataframe
    df = dataframe.copy()
    df['PROD_DIFF'] = X_test['PROD_DIFF']
    if not living:
        df.drop(to_drop, axis=1)

    # create path for plots
    decision_path = Path(results_path) / "decision_boundaries"
    decision_path.mkdir(parents=True, exist_ok=True)

    # create path for shape files
    final_maps = Path(results_path) / "final_maps"
    final_maps.mkdir(parents=True, exist_ok=True)

    # create path for boxplots
    box_path = Path(results_path) / "boxplots"
    box_path.mkdir(parents=True, exist_ok=True)

    print(f'THESE ARE THE FEATURES: {features}')
    c = set(combinations(features, 2))
    combos = list(c)

    for comb in combos:
        # print(f'the comb is a {type(comb)} and it is this: {comb}')
        comb = list(comb)
        # print(f'the comb is a {type(comb)} and it is this: {comb}')

        train = X_train[comb].to_numpy()
        test = X_test[comb].to_numpy()
        num_points = len(test)
            
        
        x_min, x_max = test[:, 0].min() - 5, test[:, 0].max() + 5
        y_min, y_max = test[:, 1].min() - 5, test[:, 1].max() + 5

        for param in params:
                
            clf = OneClassSVM(**param).fit(train)            
            y = clf.predict(test)

            decision_bounds = []
        
            print(f'Model parameters: {clf.get_params()}')
            if not living:
                count = np.count_nonzero(y == 1)                    
                # print(f'The number of predicted dead trees is: {count}')
            else:
                count = np.count_nonzero(y == -1)                    
                # print(f'The number of predicted living trees is: {count}')

                
            acc = (count/num_points)*100

            if int(acc) >= 65:
                # add predictions to geodataframe
                print(f'Saving model predictions with {param} to dataframe. Accuracy: {acc}\n Features: {comb}')
                df[str(comb[0][0])+'_'+str(comb[1][0])+'_'+str(list(param.values())[0][:2])+'_'+str(list(param.values())[1][:2])+str(list(param.values())[2])[:4]] = y
                # df[str(comb[0])+'_'+str(comb[1])] = y

                # Settings for plotting
                _, ax = plt.subplots(figsize=(10, 9))
            
                ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

                # Plot decision boundary and margins
                common_params = {"estimator": clf, "X": test, "ax": ax}

                disp = DecisionBoundaryDisplay.from_estimator(
                **common_params,
                response_method="predict",
                plot_method="pcolormesh",
                alpha=0.3,
            )
                
                wdisp = DecisionBoundaryDisplay.from_estimator(
                **common_params,
                response_method="decision_function",
                plot_method="contour",
                levels=[-1, 1],
                colors=["k", "k"],
                linestyles=["-", "--"]
            )
            
                    
                for item in wdisp.surface_.collections:
                    for i in item.get_paths():
                        decision_bounds.append(i.vertices)

                # Plot bigger circles around samples that serve as support vectors
                ax.scatter(
                clf.support_vectors_[:, 0],
                clf.support_vectors_[:, 1],
                s=80,
                facecolors="none",
                edgecolors="w",
            )
                # Plot samples by color and add legend
                scatter = ax.scatter(test[:, 0], test[:, 1], s=40, c=y, label=y, edgecolors="k")
                ax.scatter(test[:, 0], test[:, 1], c=y, s=50, edgecolors="k")
                ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
                if not living:
                    ax.set_title(f"Decision boundaries of {param['kernel']} kernel in OCSVC using {param} on {year} data.\n The number of predicted dead trees is: {count}")     
                else:
                    ax.set_title(f"Decision boundaries of {param['kernel']} kernel in OCSVC using {param} on {year} data.\n The number of predicted living trees is: {count}")        
                
                ax.set_xlabel(comb[0])
                ax.set_ylabel(comb[1])
            
                plt.savefig(Path(decision_path, 'dec_bound_%s_%s_%s_%s.png' % (str(comb[0]), str(comb[1]), str(param).replace(':', '_'),year), dpi=150))

                #if int(acc) > 90:
                    # create and save boxplots
                    #create_boxplots(df, param, comb, year, box_path, prod='TPROD')
                    
    # df.to_excel(Path(results_path, "model_predictions_%s.xlsx" % year))
    df.to_file(Path(final_maps, "two_feature_predictions_for_%s.shp" % year))

    #_ = plt.show()
    # return the decision boundaries for each combination
    print(f'This is the number of decision bounds: {len(decision_bounds)}')
    return decision_bounds

def evaluate_models(best_params, X_train, X_test, X_living, dataframe, results_path, year, phen_vars=PHEN_VARS, to_drop=TO_DROP, score_functions=[f_classif, mutual_info_classif], dead_data=True):

    # phenology variables
    phen = phen_vars.copy()
    phen.append('PROD_DIFF')

    # create subdirectories
    plot_path = Path(results_path) / "feature_plotting"
    plot_path.mkdir(parents=True, exist_ok=True)

    map_path = Path(results_path) / "initial_maps"
    map_path.mkdir(parents=True, exist_ok=True)

    # create dataframes for exports
    df = dataframe.copy().drop(to_drop, axis=1)
    feature_df = pd.DataFrame(columns = ['feature', 'score'])

    # features_for_boundaries = []

    # create empty dictionary for models    
    validated_models = {}
    num = 1

    for param in best_params:
        for func in score_functions:
            print(f'Evaluating score function: {str(func)}')
            m = {}
            m['params'] = param
            model = OneClassSVM(**param)
            print(' ')
            print('Training model with: ')
            print(' %s' % (param))
            model.fit(X_train)

            # predict for dead data
            pred = model.predict(X_test)            

            # predict for living data
            live_pred = model.predict(X_living)

            # convert the pred numpy array into pandas series for plotting tone
            tone = pd.Series(pred)

            # dead trees/live trees counting
            if dead_data:
                count_dead = np.count_nonzero(pred == 1)
                print(f'Number of trees classified as dead: {count_dead}\n The total number of dead trees is: {len(pred)}')

            dead_data = False

            if not dead_data:
                count_living = np.count_nonzero(live_pred == -1)
                print(f'Number of trees classified as living: {count_living}\n The total number of living trees is: {len(live_pred)}')
            acc = (count_dead / len(pred)) * 100            
            print(f'The accuracy-score of model number {num} is: {acc}')
            live_acc = (count_living / len(live_pred)) * 100
            print(f'The living accuracy-score of model number {num} is: {live_acc}')

            dead_data = True

            # add model, score function and accuracy to feature excel sheet
            model_row = {'feature': str(model), 'score': "Score function: " + "__" + str(func) + "__" + "dead acc: " + str(acc) + "__" + "living acc: " + str(live_acc)}
            feature_df = feature_df._append(model_row, ignore_index=True)

            feature_importance = SelectKBest(score_func=func, k='all')
            feature_importance.fit(X_test, pred)
            fsr = pd.DataFrame({'feature': phen,
                                'score':feature_importance.scores_,}).sort_values(by=['score'], ascending=False)
            # Create list of features - ALL
            for_bounds = list(fsr.sort_values(by=['score'], ascending=False)[:14].iloc[:, 0])
            print(for_bounds)
            features_for_boundaries = for_bounds

            feature_df = feature_df._append(fsr, ignore_index=True)
            
            print("The accuracy of model number %d is: %d percent" % (num, acc))
            validated_models[model] = {'accuracy': acc, 
                                         'feature_importance': fsr[['feature', 'score']], 
                                         'predictions': pred,
                                         'score function': func}
            

            # plot feature importances
            # with PdfPages(Path(results_path, '%s_feature_importance_%s.pdf' % (str(model)[11:], str(func)[10:30]))) as pdf:
            with PdfPages(Path(plot_path, '%s_feature_importance_%s.pdf' % (str(model)[11:], str(func)[10:30]))) as pdf:
                fig, ax=plt.subplots(figsize=(5, 5))
                plt.title('Feature Importances for %s\n' % str(func)[10:30], fontsize=12)
                plt.xlabel('Score', fontsize=8)
                plt.ylabel('Features', fontsize=8)
                plt.tick_params(axis='both', which='major', labelsize=6)
                plot=sns.barplot(data=fsr, 
                             x='score', 
                             y='feature', 
                             palette='viridis',
                             linewidth=0.5, 
                             saturation=2, 
                             orient='h',
                             ax=ax)
                pdf.savefig(fig)
            
            num += 1
            df[str(model)] = pred
    
    df.to_excel(Path(results_path, "model_predictions_%s.xlsx" % year))
    df.to_file(Path(map_path, "predictions_for_%s.shp" % year))
    feature_df.to_excel(Path(results_path, "model_features_%s.xlsx" % year))
    print(f'the number of validated models is: {len(validated_models)}')
    print(f'THIS IS THE LENGTH OF features_for_boundaries: {len(features_for_boundaries)}')
    return validated_models, features_for_boundaries

def create_boxplots(dataset, param, model, year, box_path,prod='TPROD'):
    col = str(model[0])+'_'+str(model[1])
    data = [dataset[dataset[col] == -1][prod], dataset[dataset[col] == 1][prod]]

    fig = plt.figure(figsize =(10, 7))
    
    ax = fig.add_axes([0, 0, 1, 1])
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(data, patch_artist = True,
                    notch ='True', vert = 1)

    colors = [
            #'#0000FF', 
            '#00FF00', 
            '#FFFF00', 
            #'#FF00FF'
            ]

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # changing color and linewidth of whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linewidth = 1.5,
                    linestyle =":")

    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B',
                linewidth = 2)

    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red',
                linewidth = 3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                color ='#e7298a',
                alpha = 0.5)
        
    # x-axis labels
    ax.set_xticklabels(['Living', 'Dead',])

    # Adding title 
    plt.title("Total Productivity for %s dataset" % year)

    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    # show plot
    # plt.show()
    # save boxplot
    plt.savefig(Path(box_path, 'boxplots_%s_%s_%s.png' % (str(model) , str(param).replace(':', '_'), year), dpi=100))
