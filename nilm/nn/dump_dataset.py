from nilmtk import DataSet
import numpy as np

def dumpDataset():
    test = DataSet('dataset/ukdale/ukdale.h5')
    building = 1  ## 选择家庭house
    test.set_window(start="18-03-2013") ## 2013年3月18号之后的作为数据集
    test_elec = test.buildings[building].elec

    gt= {}
    for meter in test_elec.submeters().meters:
        gen = next(meter.load())
        values = gen.values
        index = gen.index
        label = meter.label()
        i = 0;
        while(1):
            name = label+str(i)
            if name not in gt:
                break
            else:
                i += 1
        np.save('dataset/ukdale/UKData/'+name,values)
        gt[name] = 1
        print(values.shape,'   ',index.shape)
        print('saving...', name)

if __name__ == "__main__":
    dumpDataset()