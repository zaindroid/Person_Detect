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
from metrics_refbox_msgs.msg import PersonDetectionResult
from metrics_refbox_msgs.msg import Command
from mbs_msgs.msg import FloatArray
import copy
import numpy as np
# body_estimation = Body('../model/body_pose_model.pth')
body_estimation = Body('/home/zainey/competition/comp_ws/src/hbrs_dev/person_detect/model/body_pose_model.pth')

hand_estimation = Hand('/home/zainey/competition/comp_ws/src/hbrs_dev/person_detect/model/hand_pose_model.pth')



class PersonDetector:
    def __init__(self):
        self.bounding_box = 0
        self.img=0
        self.clip_size=5
        self.image_queue=None
        self.bb_pub = rospy.Publisher("/metrics_refbox_client/person_detection_result", PersonDetectionResult, queue_size=10)
        # self.number_subscriber = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw", Image, self._input_image_cb)
        self.referee_command_sub = rospy.Subscriber("/metrics_refbox_client/command", Command, self._referee_command_cb)
        rospy.loginfo("Waiting for referee box to be ready...")
        self.stop_sub_flag=False
        self.cv_bridge=CvBridge()
        self.person_flag=False
        
        
    def show_skeleton(self, test_image):
        test_image = '/home/zainey/competition/comp_ws/src/hbrs_dev/person_detect/scripts/2.jpg'
        self.img = cv2.imread(test_image)  # B,G,R order
        x=0
        y=0
        maxy_x=0
        maxy_y=0
        candidate, subset = body_estimation(self.img)
        canvas = copy.deepcopy(self.img)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        if subset.size!=0:
            self.person_flag=True
        
        if self.person_flag:
            image_height = self.img.shape[0]
            image_width = self.img.shape[1]
            off=5
            
            for person in subset.astype(int):
                has_left = np.sum(person[[5, 6, 7]] == -1) == 0
                has_right = np.sum(person[[2, 3, 4]] == -1) == 0
                if not (has_left or has_right):
                    continue
                else:
                    right_wrist_index, left_wrist_index,right_foot_index, left_foot_index,nose_index,neck_index = person[[4, 7, 10,13,0,1]]
                    right_elbow_index,left_elbow_index,right_knee_index, left_knee_index=person[[3,6,9,12]]
                    right_shoulder_index,left_shoulder_index=person[[2,5]]
                    x1, y1 = candidate[nose_index][:2]
                    x2, y2 = candidate[right_wrist_index][:2]
                    x3, y3 = candidate[right_foot_index][:2]
                    x4, y4 = candidate[left_foot_index][:2]
                    x5, y5 = candidate[left_wrist_index][:2]
                    x6, y6 = candidate[neck_index][:2]
                    x7, y7 = candidate[right_elbow_index][:2]
                    x8, y8 = candidate[left_elbow_index][:2]
                    x9, y9 = candidate[right_knee_index][:2]
                    x10, y10 = candidate[left_knee_index][:2]
                    x11, y11 = candidate[right_shoulder_index][:2]
                    x12, y12 = candidate[left_shoulder_index][:2]
                    
                    ## shift fixes
                    y1=y1+0.7*(y6-y5)
                    x2=x2 + 1*(x2-x7)
                    y2=y2 + 1*(y2-y7)
                    
                    x5=x5 + 1*(x5-x8)
                    y5=y5 + 1*(y5-y8)
                    
                    x3=x3 + 2*(x3-x9)
                    y3=y3 + 0.4*(y3-y9)
                    
                    x4=x4 + 0.3*(x4-x10)
                    y4=y4 + 0.9*(y4-y10)
                

            
                    
                    convex_hull=[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8],[x9,y9],[x10,y10],[x11,y11],[x12,y12]]
            #         convex_hull=[[x1,y1-(0.4*(y6-y5))],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]
                    convex_hull=np.array(convex_hull)
                    min_x=int(convex_hull[0][0])
                    min_y=int(convex_hull[0][1])
                    max_x=int(convex_hull[0][0])
                    max_y=int(convex_hull[0][1])
                    for x,y in convex_hull:
                        if x<min_x: min_x=int(x)
                        if y<min_y: min_y=int(y)
                        if x>max_x: max_x=int(x)
                        if y>max_y: max_y=int(y)
                    


                    w=max_x-min_x
                    h=max_y-min_y
                    maxy_x=min_x+w
                    maxy_y=min_y+h
                    if ( maxy_x)>image_width:
                        maxy_x=image_width
                    if ( maxy_y)>image_height:
                        maxy_y=image_height
                        

                    cv2.rectangle(self.img, (int(min_x-off), int(min_y-off)), (int(maxy_x), int(maxy_y)), (0, 0,255), 2, lineType=cv2.LINE_AA)
                    # plt.imshow(canvas[:, :, [2, 1, 0]])
                    # cv2.imwrite('jjimage.jpg',self.img)
                    # cv2.imshow("out.jpg",self.img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

            # detect hand
            # hands_list = util.handDetect(candidate, subset, self.img)
    
            # all_hand_peaks = []
            # for x, y, w, is_left in hands_list:
            #     cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            #     cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
            return self.person_flag,[x, y, maxy_x, maxy_y] 
        else:
            return 0

    def detector(self):
        self.img= self.image_queue[4]

        # new_msg = IntArray()
        # logic for bounding box goes here.....
        output=self.show_skeleton(self.img)
        result = PersonDetectionResult()
        result.message_type = result.RESULT
        
        if output!=0:
            self.bounding_box=output
            result.person_found = True
            # img = cv2.imread(os.path.join(self.sample_images_path, 'image.jpg'), cv2.IMREAD_COLOR)
            result.image = self.cv_bridge.cv2_to_imgmsg(self.img, encoding='passthrough')
            print(self.bounding_box)
            result.box2d.min_x = int(self.bounding_box[1][0])
            result.box2d.max_x = int(self.bounding_box[1][2])
            result.box2d.max_y = int(self.bounding_box[1][3])
            result.box2d.min_y = int(self.bounding_box[1][1])
            self.bb_pub.publish(result)

        else:
            self.bounding_box=output
            result.person_found = False
            result.image = self.cv_bridge.cv2_to_imgmsg(self.img, encoding='passthrough')
            self.bb_pub.publish(result) 

        
        return self.bounding_box

    def _input_image_cb(self, msg):
        """
        :msg: sensor_msgs.Image
        :returns: None
        """
        try:
            # if not self.stop_sub_flag:
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
                # cnt = 0
                # for i in self.image_queue:
                #     cv2.imwrite('/home/zainey/competition/comp_ws/src/hbrs_dev/person_detect/results/images' + str(cnt) + '.jpg',i)
                #     cnt+=1
                #     # cv2.imshow(i)
                    # cv2.waitKey(50)
                rospy.loginfo("Input images saved on local drive")
                # call object inference method
                # print("Image queue size: ", len(self.image_queue))
                # waiting for referee box to be ready
                
                self.image_sub.unregister()
        
                result= self.detector()
                self.stop_sub_flag = False
                print(result)
        # else:
            #     print("Clip size reached")

        except CvBridgeError as e:
            rospy.logerr("Could not convert ros sensor msgs Image to opencv Image.")
            rospy.logerr(str(e))
            self._check_failure()
            return

    def _referee_command_cb(self, msg):
        
        # Referee comaand message (example)
        '''
        task: 1
        command: 1
        task_config: "{\"Target object\": \"Cup\"}"
        uid: "0888bd42-a3dc-4495-9247-69a804a64bee"
        '''

        # START command from referee
        if msg.task == 2 and msg.command == 1:

            print("\nStart command received")

            # set of the HSR camera to get front straight view
            # if not self.move_front_flag:
            #     self._hsr_head_controller('front')

            # start subscriber for image topic
            self.image_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw", 
                                                Image, 
                                                self._input_image_cb)

           
            print("\n")
            print("Initiating person detection  ")
            print("\n")
        
        # STOP command from referee
        if msg.command == 2:
            self.stop_sub_flag = True
            self.image_sub.unregister()
            
            rospy.loginfo("Referee Subscriber stopped")

    
if __name__ == '__main__':
    rospy.init_node('person_detect')
    PersonDetector()
    rospy.spin()

    