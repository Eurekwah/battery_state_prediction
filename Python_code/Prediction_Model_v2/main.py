import Data 
import Soc_v2 as model
import Soc_test_v2 as test



Data.get_data()

pre_data, test_data = model.get_pre()
model.train(pre_data)

for i in pre_data:
    data_X, data_Y = model.get_Data(i)
    test.test(data_X, data_Y)

test_x, test_y = model.get_Data(test_data)
test.test(test_x, test_y)


