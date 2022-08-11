from dl_tools.superres_models import RedNetModel, SRCNNModel

rednet = RedNetModel((128, 128, 1), filters=8).build_model()

srcnn = SRCNNModel((128, 128, 1), filters=64).build_model()
print(srcnn.summary())