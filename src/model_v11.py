"""
blend model v0 and model v1
"""


def get_model():
    # init models
    import model_v0
    import model_v1
    from blender import BlendingClassifier
    modelV0 = model_v0.get_model()
    modelV1 = model_v1.get_model()
    # init weights
    weights = [0.5, 0.5]
    return BlendingClassifier(models=[modelV0, modelV1], weights=weights)


def transform(df_text):
    import model_v1
    return model_v1.transform(df_text)
