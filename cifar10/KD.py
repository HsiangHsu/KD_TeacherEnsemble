import tensorflow as tf
from data import get_cifar10_data
import teacher
import student
from metrics import accuracy
from time import localtime, strftime

train_x, train_y, valid_x, valid_y, test_x, test_y = get_cifar10_data()
number_of_classes = test_y.shape[1]


teacher_config = {}
teacher_config['max_epochs'] = 10
teacher_config['batch_size'] = 50
teacher_config['num_classes'] = number_of_classes

student_config = {}
student_config['max_epochs'] = 50
student_config['batch_size'] = 50
student_config['num_classes'] = number_of_classes
student_config['temperature'] = 1.5

teacher = teacher.DeepModel(train_x, number_of_classes)
student = student.DeepModel(train_x, number_of_classes, student_config['temperature'])

def callback_fn(session, config, model):
    model_accuracy = accuracy(valid_x, valid_y, config['batch_size'], config['num_classes'], session, model) 
    print(f"Validation set accuracy: {model_accuracy * 100.}")
    file.write("Validation set accuracy: {}\n".format(model_accuracy * 100))
    file.flush()


filename = 'train_teacher'
file = open(filename+'_log.txt','w')
file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()

saver = tf.train.Saver()
teacher_list = []
number_of_teacher=10
noise_sigma_list = [0, .1, .2, .5, 1, 2, 5]
file.write('Training\n')
file.flush()
for i in range(number_of_teacher):
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    teacher.train(train_x, train_y, session, teacher_config, callback_fn)
    saver.save(session, "models/model_teacher"+str(i))
    for j in range(len(noise_sigma_list)):
        sigma = noise_sigma_list[j]
        teacher_accuracy = accuracy(test_x, test_y, teacher_config['batch_size'], teacher_config['num_classes'], session, teacher, sigma)
        print("Noise Variance: {}, Teacher network test set accuracy: {}\n".format(sigma, teacher_accuracy))
        file.write("Teacher {} \n Noise Variance: {},  Teacher network test set accuracy: {}\n".format(i, sigma, teacher_accuracy))
        file.flush()
    teacher_list.append(teacher)

train students
student.train(train_x, session, student_config, [teacher_list[0]], callback_fn)
for j in range(len(noise_sigma_list)):
        sigma = noise_sigma_list[j]
        student_accuracy = accuracy(test_x, test_y, student_config['batch_size'], student_config['num_classes'], session, student, sigma)
        file.write("\n\n\n Student {} \n Noise Variance: {},  Teacher network test set accuracy: {}\n".format(i, sigma, student_accuracy))
        file.flush()
        print("Noise Variance: {}, Student network test set accuracy: {}\n".format(sigma, student_accuracy))

file.write("(1 teachers) Student network test set accuracy: {}\n".format(student_accuracy))
file.flush()        
print(f"(1 teacher) Student network test set accuracy: {student_accuracy}")
saver.save(session, "models/model_student1")

student.train(train_x, session, student_config, teacher_list[0:3], callback_fn)
for j in range(len(noise_sigma_list)):
        sigma = noise_sigma_list[j]
        student_accuracy = accuracy(test_x, test_y, student_config['batch_size'], student_config['num_classes'], session, student, sigma)
        file.write("Student {} \n Noise Variance: {},  Teacher network test set accuracy: {}\n".format(i, sigma, student_accuracy))
        file.flush()
        print("Noise Variance: {}, Student network test set accuracy: {}\n".format(sigma, student_accuracy))
        
file.write("(3 teachers) Student network test set accuracy: {}\n".format(student_accuracy))
file.flush()
print(f"(3 teachers) Student network test set accuracy: {student_accuracy}")
saver.save(session, "models/model_student3")

student.train(train_x, session, student_config, teacher_list[0:5], callback_fn)
for j in range(len(noise_sigma_list)):
        sigma = noise_sigma_list[j]
        student_accuracy = accuracy(test_x, test_y, student_config['batch_size'], student_config['num_classes'], session, student, sigma)
        file.write("Student {} \n Noise Variance: {},  Teacher network test set accuracy: {}\n".format(i, sigma, student_accuracy))
        file.flush()
        print("Noise Variance: {}, Student network test set accuracy: {}\n".format(sigma, student_accuracy))
        
file.write("(5 teachers) Student network test set accuracy: {}\n".format(student_accuracy))
file.flush()
print(f"(5 teachers) Student network test set accuracy: {student_accuracy}")
saver.save(session, "models/model_student5")



student.train(train_x, session, student_config, teacher_list, callback_fn)
for j in range(len(noise_sigma_list)):
        sigma = noise_sigma_list[j]
        student_accuracy = accuracy(test_x, test_y, student_config['batch_size'], student_config['num_classes'], session, student, sigma)
        file.write("Student {} \n Noise Variance: {},  Teacher network test set accuracy: {}\n".format(i, sigma, student_accuracy))
        file.flush()
        print("Noise Variance: {}, Student network test set accuracy: {}\n".format(sigma, student_accuracy))
        
file.write("(10 teachers) Student network test set accuracy: {}\n".format(student_accuracy))
file.flush()
print(f"(10 teachers) Student network test set accuracy: {student_accuracy}")
saver.save(session, "models/model_student10")

file.write("... completed!\n")
file.flush()
file.close()