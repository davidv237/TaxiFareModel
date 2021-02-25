# imports
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data,clean_data
from TaxiFareModel.utils import compute_rmse
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), StandardScaler())

        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        preprocessor = ColumnTransformer([('time', pipe_time, time_cols),
                                      ('distance', pipe_distance, dist_cols)]
                                      )
        # Add the model of your choice to the pipeline
        final_pipe = Pipeline(steps=[('preprocessing', preprocessor),
                                ('regressor', LinearRegression())])

        # display the pipeline with model
        self.pipeline = final_pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        print(compute_rmse(y_pred,y_test))
        return compute_rmse(y_pred,y_test)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    clean_df = clean_data(df)
    # set X and y
    y = clean_df['fare_amount']
    X = clean_df.drop(columns=['fare_amount'])
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    # train
    trainer = Trainer(X_train,y_train)
    trainer.set_pipeline()
    trainer.run()
    trainer.evaluate(X_val,y_val)


    # evaluate
