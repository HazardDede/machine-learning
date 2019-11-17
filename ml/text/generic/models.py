
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def model_space(
    enable_sgd=True, enable_mnb=False, enable_rforest=False, enable_svc_linear=False,
    enable_svc_nonlinear=False, enable_gboost=False
):
    space = []
    
    if enable_sgd:
        # SGD classifier
        sgd_model = {
            'clf': (SGDClassifier(random_state=42), ),
            'p3': (None,),
            'clf__alpha': (1e-3, 1e-4),
            'clf__loss': ('hinge', 'log', 'modified_huber')
        }
        space.append(sgd_model)
    
    if enable_mnb:
        # MultinomialNB classifier
        mnb_model = {
            'clf': (MultinomialNB(), ),
            'p3': (None,),
            'clf__alpha': np.linspace(0.5, 1.5, 6),
            'clf__fit_prior': [True, False],
        }
        space.append(mnb_model)

    if enable_rforest:
        # RandomForest classifier
        rforest_model = {
            'clf': (RandomForestClassifier(random_state=42), ),
            'clf__n_estimators': (100,),  # (100, 1000, 5000),
            'clf__criterion': ('gini', 'entropy'),  # ('gini', 'entropy')
            'clf__max_depth': (5, 10, None),
            'clf__max_features': ('auto', 'log2', None)
        }
        space.append(rforest_model)

    if enable_svc_linear:
        # SVC linear kernel
        svc_lin_model = {
            'clf': (SVC(),),
            'clf__C': [0.1, 1, 10, 100, 1000],
            'clf__kernel': ['poly'],
            'clf__degree': [1, 2, 3],
            'clf__gamma': ('scale', )  # Get rid of the deprecation warning -> unused
        }
        space.append(svc_lin_model)
    
    if enable_svc_nonlinear:
        # SVC non-linear kernel
        svc_nonlin_model = {
            'clf': (SVC(),),
            'clf__C': [0.1, 1, 10, 100, 1000],
            'clf__kernel': ['rbf'],
            'clf__gamma': ['auto', 'scale', 0.1, 1, 10, 100]
        }
        space.append(svc_nonlin_model)

    if enable_gboost:
        # GBOOST
        gboost_model = {
            'clf': (GradientBoostingClassifier(random_state=42),),
            'clf__n_estimators': (100, 1000),
            'clf__learning_rate': (0.1, 0.2, 0.5)
        }
        space.append(gboost_model)

    if not space:
        raise RuntimeError("You have to enable at least one model.")


    return space
