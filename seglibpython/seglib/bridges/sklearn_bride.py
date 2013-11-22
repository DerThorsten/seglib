




def _imageSklearn(cls):

    def toImgReshape(X):
        if X is None:
            return None

    def fromImgReshape(X):
        if X is None:
            return None

    class ImageSklearn(object):
        def __init__(self,*args,**kwargs):

            # argument
            self.args   = args
            self.kwargs = kwargs

            # class (should be sklearn compatible)
            self.cls    = cls
            self.clsObj = self.cls(*self.args,**self.kwargs)




        def fit(X,y=None):                    
            """fit"""
            return self.clsOb.fit(X=fromImgReshape(X),y=fromImgReshape(y)):
        def fit_predict(X):                  
            """fit and predict"""
            return toImgReshape(self.clsOb.fit_predict(X=fromImgReshape(X))):
        def fit_transform(X, y=None):            
            """Compute transformation"""
            return toImgReshape(self.clsOb.fit_transform(X=fromImgReshape(X),y=fromImgReshape(y))):
        def get_params(deep=None):               
            """Get parameters for this estimator."""
            return self.clsObj.get_params(deep=deep)
        def predict(X):                       
            """predict."""
            return toImgReshape(self.clsObj.predict(X=fromImgReshape(X)))
        def score(X):                         
            """core of learer"""
            return self.clsObj.score(X=fromImgReshape(X))
        def set_params(**params):             
            self.clsObj.set_params(**params)
        def transform(X,y=None):              
            """transform """
            return  toImgReshape(X=self.clsObj.transform(fromImgReshape(X),y=fromImgReshape(y)))
