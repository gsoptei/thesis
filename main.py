from pathlib import Path
from datetime import datetime
from json import dumps
from utils import *
from settings import *

# setup
base_year = BASE_YEAR
validation_year = LIVING_YEAR
desc_string = "gergely_ocsvm_entire_data"

# load points for training data
# insert your own path
data_file = Path('' % base_year)

results_path = Path("%s_res_%s" % (datetime.now().strftime("%Y%m%d-%H%M%S"), desc_string))
results_path.mkdir(parents=True, exist_ok=True)

# load data for training
training_points_gdf = gpd.read_file(data_file)
train_gdf = extract_raster_values_at_points(base_year, RASTERS, training_points_gdf, PHEN_VARS)

# format training data
train_nulls_dropped = drop_nulls(train_gdf)
train_dead = add_dead(train_nulls_dropped)
train_formatted = format_dates(train_dead , base_year)
X_train, y_train, train_subsetted = subset_data(train_formatted)

# load living tree data for further validation
living_points_gdf = gpd.read_file(VALIDATION_POINTS)
living_gdf = extract_raster_values_at_points(validation_year, RASTERS, living_points_gdf, PHEN_VARS)

# format living data
living_nulls_dropped = drop_nulls(living_gdf)
living_dead = add_dead(living_nulls_dropped, dead=False)
living_formatted = format_dates(living_dead , validation_year)
X_living, y_living, living_subsetted = subset_data(living_formatted)

# Get hyperparameters from training data to valuidate in main()
models_to_validate = search_hyperparameters(SVM_PARAMS, X_train, y_train, X_living, y_living, results_path)


def main(year, results_path=results_path):
    # TODO: use proper nomenclature for 'test' and 'validation' sets
    # create results directory
    year_path = Path(results_path) / ("%s_data" % year)
    year_path.mkdir(parents=True, exist_ok=True)


    # open results description file
    results_desc_file = Path(year_path, "description_%s.txt" % year).open('w')

    # open the data
    results_desc_file.write('Using the points in the following shape for model training: %s\n' % str(data_file))
    results_desc_file.write(f'Phenology variables: {PHEN_VARS} \n')

    # load data for testing
    testing_points_gdf = gpd.read_file('path/to/your/points/points_%s.shp' % year)
    test_gdf = extract_raster_values_at_points(year, RASTERS, testing_points_gdf, PHEN_VARS)

    # format test data
    test_nulls_dropped = drop_nulls(test_gdf)
    test_dead = add_dead(test_nulls_dropped)
    test_formatted = format_dates(test_dead, year)
    X_test, y_test, test_subsetted = subset_data(test_formatted)

    # load live tree validation data
    validation_data = extract_raster_values_at_points(validation_year, RASTERS, VALIDATION_GDF, PHEN_VARS)

    # format validation data
    val_nulls_dropped = drop_nulls(validation_data)
    val_formatted = format_dates(val_nulls_dropped, validation_year)
    
    # search for hyperparameters - test with test dataset
    results_desc_file.write("Search configuration: %s\n" % dumps(SVM_PARAMS, indent=4))

    
    results_desc_file.write("Number of models to validate: %d \n" % len(models_to_validate))
    results_desc_file.write("Models to validate: \n%s\n" % dumps(str(models_to_validate), indent=4))

    validated_models, features_for_boundaries = evaluate_models(models_to_validate,
                                                                X_train,
                                                                X_test,
                                                                X_living,
                                                                test_subsetted,
                                                                year_path,
                                                                year)

    results_desc_file.write("\nModels validated on validation data: \n%s\n" % dumps(str(validated_models), indent=4))
    results_desc_file.write("\nFeaturues for plotting decision boundaries: \n%s\n" % str(features_for_boundaries))

    # decision boundaries for the dead datasets
    dec_bound = plot_decision_boundaries(train_formatted, 
                                        test_formatted,
                                        year_path,
                                        year,
                                        features_for_boundaries, 
                                        models_to_validate,
                                        PHEN_VARS,
                                        test_formatted)

    results_desc_file.write("\nLenght of decision boundaries for dead data: \n%d\n" % len(dec_bound))
    # create new directories and plot living data decision bounds for each year
    living_path = Path(results_path / ("live_decision_boundaries_with_%s_data" % year))
    living_path.mkdir(parents=True, exist_ok=True)

    # decision boundaries for 'living' datasets
    dec_bound_living = plot_decision_boundaries(train_formatted,
                                                living_formatted,
                                                living_path, 
                                                'living',
                                                features_for_boundaries,
                                                models_to_validate,
                                                PHEN_VARS,
                                                living_formatted,
                                                living=True)
    
    results_desc_file.write("\nLenght of decision boundaries for living data: \n%d\n" % len(dec_bound_living))

if __name__ == "__main__":
    for i in YEARS:
        main(i)
