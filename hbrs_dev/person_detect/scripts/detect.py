#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt
from person_detect.hand import Hand
from person_detect.body import Body
from person_detect import model
from person_detect import util
from metrics_refbox_msgs.msg import ObjectDetectionResult
from mbs_msgs.msg import FloatArray
import copy
body_estimation = Body('../model/body_pose_model.pth')
hand_estimation = Hand('../model/hand_pose_model.pth')



class PersonDetector:
    def __init__(self):
        self.bounding_box = 0
        self.img=0
        self.clip_size=5
        self.image_queue=None
        self.pub = rospy.Publisher("/metrics_refbox_client/object_detection_result", ObjectDetectionResult, queue_size=10)
        self.number_subscriber = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw", Image, self._input_image_cb)
        self.stop_sub_flag=False
        self.cv_bridge=CvBridge()
        
        
    def show_skeleton(self, test_image):
        # test_image = 'images/demo.jpg'
        # self.img = cv2.imread(test_image)  # B,G,R order
        candidate, subset = body_estimation(self.img)
        canvas = copy.deepcopy(self.img)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        # detect hand
        hands_list = util.handDetect(candidate, subset, self.img)

        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # if is_left:
                # plt.imshow(self.img[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
                # plt.show()
            # peaks = hand_estimation(self.img[y:y+w, x:x+w, :])
            # peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            # peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            # # else:
            #     peaks = hand_estimation(cv2.flip(self.img[y:y+w, x:x+w, :], 1))
            #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
            #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            #     print(peaks)
            # all_hand_peaks.append(peaks)

        # canvas = util.draw_handpose(canvas, all_hand_peaks)

        # plt.imshow(canvas[:, :, [2, 1, 0]])
        # plt.axis('off')
        # plt.show()
        
        # return canvas
        return [x, y, x+w, y+w] 


    def detector(self):
        self.img= self.image_queue[4]
        # new_msg = IntArray()
        # logic for bounding box goes here.....
        self.bounding_box=self.show_skeleton(self.img)
        # new_msg.data = self.bounding_box
        # self.pub.publish(new_msg)
        

        result = ObjectDetectionResult()
        result.message_type = result.RESULT
        result.result_type = ObjectDetectionResult.BOUNDING_BOX_2D
        result.object_found = True
        # img = cv2.imread(os.path.join(self.sample_images_path, 'image.jpg'), cv2.IMREAD_COLOR)
        result.image = self.cv_bridge.cv2_to_imgmsg(self.img, encoding='passthrough')
        result.box2d.min_x = self.bounding_box[0]
        result.box2d.min_y = self.bounding_box[1]
        result.box2d.max_x = self.bounding_box[2]
        result.box2d.max_y = self.bounding_box[3]
        self.pub.publish(result)
        # self.result_publishers['object_detection'].publish(result)
        return self.bounding_box

    def _input_image_cb(self, msg):
        """
        :msg: sensor_msgs.Image
        :returns: None
        """
        try:
            if not self.stop_sub_flag:
                rospy.loginfo("Image received..")
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                if self.image_queue is None:
                    self.image_queue = []
                self.image_queue.append(cv_image)
                print("Counter: ", len(self.image_queue))
                if len(self.image_queue) > self.clip_size:
                    #Clip size reached
                    # print("Clip size reached...")
                    self.stop_sub_flag = True
                    self.image_queue.pop(0)
                    # save all images on local drive
                    cnt = 0
                    for i in self.image_queue:
                        cv2.imwrite('/home/zainey/competition/comp_ws/src/hbrs_dev/person_detect/results/images' + str(cnt) + '.jpg',i)
                        cnt+=1
                        # cv2.imshow(i)
                        # cv2.waitKey(50)
                    rospy.loginfo("Input images saved on local drive")
                    # call object inference method
                    # print("Image queue size: ", len(self.image_queue))
                    # waiting for referee box to be ready
                    rospy.loginfo("Waiting for referee box to be ready...")
                    
                    result= self.detector()
                    print(result)
            # else:
            #     print("Clip size reached")

        except CvBridgeError as e:
            rospy.logerr("Could not convert ros sensor msgs Image to opencv Image.")
            rospy.logerr(str(e))
            self._check_failure()
            return

    
if __name__ == '__main__':
    rospy.init_node('person_detect')
    PersonDetector()
    rospy.spin()