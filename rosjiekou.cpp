// Track Object---advanced by Xuancen Liu
// ----------------------------------------- 2019.9.18 at Hunan Changsha.
//  email: buaalxc@163.com
// wechat: liuxuancen003
#include <boost/thread/mutex.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <chrono>
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Pose2D.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_srvs/SetBool.h>

#include "kcftracker.hpp"

#include "amov_gimbal_control/VisionDiff.h"
#include "derror.h"
#include "kcf/WindowPosition.h"

using namespace std;
using namespace cv;

#define MARKER_SIZE 0.18
#define F1 300
#define F2 300
#define C1 320
#define C2 240

static const std::string RGB_WINDOW = "RGB Image window";

//! Camera related parameters.
int frameWidth_;
int frameHeight_;

float get_ros_time(ros::Time begin); //获取ros当前时间

std_msgs::Header imageHeader_;
cv::Mat camImageCopy_;
boost::shared_mutex mutexImageCallback_;
bool imageStatus_ = false;
boost::shared_mutex mutexImageStatus_;

DERROR derrorX, derrorY;

void cameraCallback(const sensor_msgs::ImageConstPtr &msg) {
  ROS_DEBUG("[EllipseDetector] USB image received.");

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    imageHeader_ = msg->header;
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(
          mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(
          mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

// 用此函数查看是否收到图像话题
bool getImageStatus(void) {
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

//! ROS subscriber and publisher.
image_transport::Subscriber imageSubscriber_;
image_transport::Publisher image_vision_pub;
ros::Publisher pose_pub;

cv::Rect selectRect;
cv::Point origin;
cv::Rect result;

bool select_flag = false;
bool bRenewROI = false; // the flag to enable the implementation of KCF
                        // algorithm for the new chosen ROI
bool bBeginKCF = false;
int g_control_gimbal = 1;

float get_ros_time(ros::Time begin) {
  ros::Time time_now = ros::Time::now();
  float currTimeSec = time_now.sec - begin.sec;
  float currTimenSec = time_now.nsec / 1e9 - begin.nsec / 1e9;
  return (currTimeSec + currTimenSec);
}

void bboxDrawCb(const kcf::WindowPosition::ConstPtr &msg) {
  if (msg->mode != 0) {
    selectRect.x = msg->origin_x;
    selectRect.y = msg->origin_y;
    selectRect.width = msg->width;
    selectRect.height = msg->height;
    selectRect &= cv::Rect(0, 0, frameWidth_, frameHeight_);
    if (selectRect.width * selectRect.height > 64) {
      bRenewROI = true;
    }
    g_control_gimbal = 1;
  } else {
    g_control_gimbal = 0;
  }
}

void onMouse(int event, int x, int y, int, void *) {
  if (select_flag) {
    selectRect.x = MIN(origin.x, x);
    selectRect.y = MIN(origin.y, y);
    selectRect.width = abs(x - origin.x);
    selectRect.height = abs(y - origin.y);
    selectRect &= cv::Rect(0, 0, frameWidth_, frameHeight_);
  }
  if (event == CV_EVENT_LBUTTONDOWN) {
    bBeginKCF = false;
    select_flag = true;
    origin = cv::Point(x, y);
    selectRect = cv::Rect(x, y, 0, 0);
  } else if (event == CV_EVENT_LBUTTONUP) {
    if (selectRect.width * selectRect.height < 64) {
      ;
    } else {
      select_flag = false;
      bRenewROI = true;
    }
  }
}

bool gimbalSer(std_srvs::SetBool::Request &req,
               std_srvs::SetBool::Response &resp) {
  if (req.data) {
    g_control_gimbal = 0;
  } else if (selectRect.width * selectRect.height > 0) {
    bRenewROI = true;
    g_control_gimbal = 1;
  } else {
    bRenewROI = false;
    bBeginKCF = false;
  }
  resp.success = true;
  resp.message = req.data ? "Gimbal Control Close" : "Gimbal Control Open";
  return true;
}

bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool SILENT = true;
bool LAB = false;

// Create KCFTracker object
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

int main(int argc, char **argv) {

  ros::init(argc, argv, "tracker_ros");
  ros::NodeHandle nh("~");
  image_transport::ImageTransport it(nh);
  ros::Rate loop_rate(30);
  bool auto_zoom, show_ui;
  float max_size, min_size;
  nh.param<bool>("auto_zoom", auto_zoom, false);
  nh.param<bool>("show_ui", show_ui, true);
  nh.param<float>("max_size", max_size, 0.0);
  nh.param<float>("min_size", min_size, 0.0);
  std::cout << "auto_zoom: " << auto_zoom << " "
            << "max_size: " << max_size << " "
            << "min_size: " << min_size << std::endl;

  // 接收图像的话题
  imageSubscriber_ =
      it.subscribe("/amov_gimbal_ros/gimbal_image", 1, cameraCallback);
  // 发送绘制图像
  image_vision_pub = it.advertise("/detection/image", 1);

  // diff
  ros::Publisher position_diff_pub =
      nh.advertise<amov_gimbal_control::VisionDiff>("/gimbal/track", 10);
  // ros::Publisher auto_zoom_pub =
  // nh.advertise<amov_gimbal_control::Diff>("/gimbal_server/auto_zoom", 10);
  ros::Subscriber sub_bbox_draw =
      nh.subscribe("/detection/bbox_draw", 10, bboxDrawCb);
  ros::ServiceServer server =
      nh.advertiseService("/detection/gimbal_control", gimbalSer);

  sensor_msgs::ImagePtr msg_ellipse;

  const auto wait_duration = std::chrono::milliseconds(2000);
  if (show_ui) {
    cv::namedWindow(RGB_WINDOW);
    cv::setMouseCallback(RGB_WINDOW, onMouse, 0);
  }

  float cur_time;
  float last_time;
  float last_error_x, last_error_y;
  float dt;
  float unfilter_vely, unfilter_velx;

  amov_gimbal_control::VisionDiff error_pixels;
  ros::Time begin_time = ros::Time::now();

  while (ros::ok()) {

    cur_time = get_ros_time(begin_time);
    dt = (cur_time - last_time);
    if (dt > 1.0 || dt < 0.0) {
      dt = 0.05;
    }
    while (!getImageStatus()) {
      printf("Waiting for image.\n");
      std::this_thread::sleep_for(wait_duration);
      ros::spinOnce();
    }

    Mat frame;
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(
          mutexImageCallback_);
      frame = camImageCopy_.clone();
    }
    static bool need_tracking_det = false;

    if (bRenewROI) {
      tracker.init(selectRect, frame);
      cv::rectangle(frame, selectRect, cv::Scalar(255, 0, 0), 2, 8, 0);
      bRenewROI = false;
      bBeginKCF = true;
    } else if (bBeginKCF) {
      result = tracker.update(frame);
      error_pixels.detect = 1;
      error_pixels.x = result.x + result.width / 2 - frame.cols / 2;
      error_pixels.y = result.y + result.height / 2 - frame.rows / 2;

      error_pixels.currSize = (float)result.width * (float)result.height /
                              (frameHeight_ * frameWidth_);
      error_pixels.maxSize = (float)selectRect.width *
                             (float)selectRect.height /
                             (frameHeight_ * frameWidth_);

      float error_x = error_pixels.x;
      float error_y = error_pixels.y;

      derrorX.add_error(error_x, cur_time);
      derrorY.add_error(error_y, cur_time);
      derrorX.derror_output();
      derrorY.derror_output();
      derrorX.show_error();
      derrorY.show_error();

      error_pixels.velx = derrorX.Output;
      error_pixels.vely = derrorY.Output;

      error_pixels.Ix += error_pixels.x * dt;
      error_pixels.Iy += error_pixels.y * dt;

      unfilter_velx = (error_pixels.x - last_error_x) / dt;
      unfilter_vely = (error_pixels.y - last_error_y) / dt;

      last_time = cur_time;
      last_error_x = error_pixels.x;
      last_error_y = error_pixels.y;
      // error_pixels.kp = 0.1;

      cv::rectangle(frame, result, cv::Scalar(255, 0, 0), 2, 8, 0);
    } else {
      error_pixels.detect = 0;
      error_pixels.x = 0.0;
      error_pixels.y = 0.0;
      error_pixels.Ix = 0.0;
      error_pixels.Iy = 0.0;
      error_pixels.velx = 0.0;
      error_pixels.vely = 0.0;
      error_pixels.currSize = 0.0;
      error_pixels.maxSize = 0.0;
    }
    error_pixels.kp = 0.2;
    error_pixels.ki = 0.0001;
    error_pixels.kd = 0.003;
    if (max_size != 0 && min_size != 0 && auto_zoom) {
      error_pixels.maxSize = max_size;
      error_pixels.minSize = min_size;
    }
    error_pixels.autoZoom = auto_zoom;
    error_pixels.trackIgnoreError = 35;
    if (g_control_gimbal == 0) {
      error_pixels.detect = 0;
    }
    position_diff_pub.publish(error_pixels);
    // auto_zoom_pub.publish(error_pixels);

    float left_point = frame.cols / 2 - 20;
    float right_point = frame.cols / 2 + 20;
    float up_point = frame.rows / 2 + 20;
    float down_point = frame.rows / 2 - 20;
    // draw
    line(frame, Point(left_point, frame.rows / 2),
         Point(right_point, frame.rows / 2), Scalar(0, 255, 0), 1, 8);
    line(frame, Point(frame.cols / 2, down_point),
         Point(frame.cols / 2, up_point), Scalar(0, 255, 0), 1, 8);
    putText(frame, "x:", Point(50, 60), FONT_HERSHEY_SIMPLEX, 1,
            Scalar(255, 23, 0), 3, 8);
    putText(frame, "y:", Point(50, 90), FONT_HERSHEY_SIMPLEX, 1,
            Scalar(255, 23, 0), 3, 8);

    // draw
    char s[20] = "";
    sprintf(s, "%.2f", error_pixels.x);
    putText(frame, s, Point(100, 60), FONT_HERSHEY_SIMPLEX, 1,
            Scalar(255, 23, 0), 2, 8);
    sprintf(s, "%.2f", error_pixels.y);
    putText(frame, s, Point(100, 90), FONT_HERSHEY_SIMPLEX, 1,
            Scalar(255, 23, 0), 2, 8);

    if (show_ui) {
      imshow(RGB_WINDOW, frame);
      waitKey(20);
    }

    image_vision_pub.publish(
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg());
    ros::spinOnce();
    loop_rate.sleep();
  }
}
