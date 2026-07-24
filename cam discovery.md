We'll check that later. First, we need to debug another minor issue.

Robot is running teleimager server.
I access webrtc video of robot's head camera at this address on my PC: https://192.168.123.164:60001/. The feed becomes visible
When I visit same address on meta quest, this page does not open.

When I run teleoperation and open vuer on PC at https://0.0.0.0:8012/?ws=wss://0.0.0.0:8012. Camera feed shows fine in the PC.

But when I visit same address on meta quest, vuer opens but camera feed does not show up.

(teleimager) unitree@ubuntu:~/teleimager$ teleimager-server --cf
06:10:39.916308 INFO                                                                                                                                                                    image_server.py:1371
                         ====================== Image Server Startup Guide ======================                                                                                                           
                         Please first read this repo's README.md to learn how to configure and use the teleimager.                                                                                          
                         To discover connected cameras, run the following command:                                                                                                                          
                                                                                                                                                                                                            
                             teleimager-server --cf                                                                                                                                                         
                                                                                                                                                                                                            
                         The '--cf' flag means 'camera find'.                                                                                                                                               
                         This will list all detected cameras and their details (video paths, serial numbers and physical path etc.).                                                                        
                         Use that information to fill in your 'cam_config_server.yaml' file.                                                                                                                
                         Once configured, you can start the image server with:                                                                                                                              
                                                                                                                                                                                                            
                             teleimager-server                                                                                                                                                              
                                                                                                                                                                                                            
                         Note:                                                                                                                                                                              
                          - If you have RealSense cameras, add the '--rs' flag to enable RealSense support.                                                                                                 
                          - Make sure you have proper permissions to access the camera devices (e.g., run with sudo or set udev rules).                                                                     
                         ==========================================================================                                                                                                         
06:10:42.045637 INFO     UVC driver reloaded successfully.                                                                                                                               image_server.py:535
06:10:42.872988 INFO     ======================= Camera Discovery Start ==================================                                                                               image_server.py:780
06:10:42.873553 INFO     Found video devices: ['/dev/video0', '/dev/video1', '/dev/video2', '/dev/video3', '/dev/video4', '/dev/video5']                                                 image_server.py:781
06:10:42.874169 INFO     Found RGB video devices: ['/dev/video2', '/dev/video4']                                                                                                         image_server.py:782
06:10:42.874290 INFO     ----------------------- OpenCV / UVC Camera 1 -----------------------------                                                                                     image_server.py:791
06:10:42.874610 INFO     video_path    : /dev/video2                                                                                                                                     image_server.py:792
06:10:42.874962 INFO     video_id      : 2                                                                                                                                               image_server.py:793
06:10:42.875058 INFO     serial_number : 253443064923                                                                                                                                    image_server.py:794
06:10:42.875148 INFO     physical_path : /sys/devices/platform/3610000.xhci/usb2/2-3/2-3:1.0                                                                                             image_server.py:795
06:10:42.875234 INFO     extra_info:                                                                                                                                                     image_server.py:796
06:10:42.875319 INFO         name: Intel(R) RealSense(TM) Depth Camera 435i                                                                                                              image_server.py:803
06:10:42.875401 INFO         manufacturer: Intel(R) RealSense(TM) Depth Camera 435i                                                                                                      image_server.py:803
06:10:42.875479 INFO         serialNumber: 253443064923                                                                                                                                  image_server.py:803
06:10:42.875556 INFO         idProduct: 2874                                                                                                                                             image_server.py:803
06:10:42.875632 INFO         idVendor: 32902                                                                                                                                             image_server.py:803
06:10:42.875714 INFO         device_address: 3                                                                                                                                           image_server.py:803
06:10:42.875795 INFO         bus_number: 2                                                                                                                                               image_server.py:803
06:10:42.875869 INFO         uid: 2:3                                                                                                                                                    image_server.py:803
06:10:42.977420 INFO     ----------------------- OpenCV / UVC Camera 2 -----------------------------                                                                                     image_server.py:791
06:10:42.977902 INFO     video_path    : /dev/video4                                                                                                                                     image_server.py:792
06:10:42.978066 INFO     video_id      : 4                                                                                                                                               image_server.py:793
06:10:42.978176 INFO     serial_number : 253443064923                                                                                                                                    image_server.py:794
06:10:42.978271 INFO     physical_path : /sys/devices/platform/3610000.xhci/usb2/2-3/2-3:1.3                                                                                             image_server.py:795
06:10:42.978357 INFO     extra_info:                                                                                                                                                     image_server.py:796
06:10:42.978442 INFO         name: Intel(R) RealSense(TM) Depth Camera 435i                                                                                                              image_server.py:803
06:10:42.978525 INFO         manufacturer: Intel(R) RealSense(TM) Depth Camera 435i                                                                                                      image_server.py:803
06:10:42.978602 INFO         serialNumber: 253443064923                                                                                                                                  image_server.py:803
06:10:42.978679 INFO         idProduct: 2874                                                                                                                                             image_server.py:803
06:10:42.978754 INFO         idVendor: 32902                                                                                                                                             image_server.py:803
06:10:42.978826 INFO         device_address: 3                                                                                                                                           image_server.py:803
06:10:42.978895 INFO         bus_number: 2                                                                                                                                               image_server.py:803
06:10:42.978965 INFO         uid: 2:3                                                                                                                                                    image_server.py:803
06:10:43.056453 INFO     =========================== Camera Discovery End ================================                                                                               image_server.py:815
(teleimager) unitree@ubuntu:~/teleimager$ teleimager-server --rs --cf
06:11:01.122254 INFO                                                                                                                                                                    image_server.py:1371
                         ====================== Image Server Startup Guide ======================                                                                                                           
                         Please first read this repo's README.md to learn how to configure and use the teleimager.                                                                                          
                         To discover connected cameras, run the following command:                                                                                                                          
                                                                                                                                                                                                            
                             teleimager-server --cf                                                                                                                                                         
                                                                                                                                                                                                            
                         The '--cf' flag means 'camera find'.                                                                                                                                               
                         This will list all detected cameras and their details (video paths, serial numbers and physical path etc.).                                                                        
                         Use that information to fill in your 'cam_config_server.yaml' file.                                                                                                                
                         Once configured, you can start the image server with:                                                                                                                              
                                                                                                                                                                                                            
                             teleimager-server                                                                                                                                                              
                                                                                                                                                                                                            
                         Note:                                                                                                                                                                              
                          - If you have RealSense cameras, add the '--rs' flag to enable RealSense support.                                                                                                 
                          - Make sure you have proper permissions to access the camera devices (e.g., run with sudo or set udev rules).                                                                     
                         ==========================================================================                                                                                                         
06:11:03.258806 INFO     UVC driver reloaded successfully.                                                                                                                               image_server.py:535
06:11:05.195074 INFO     ======================= Camera Discovery Start ==================================                                                                               image_server.py:780
06:11:05.195739 INFO     Found video devices: ['/dev/video0', '/dev/video1', '/dev/video2', '/dev/video3', '/dev/video4', '/dev/video5']                                                 image_server.py:781
06:11:05.197046 INFO     Found RGB video devices: []                                                                                                                                     image_server.py:782
06:11:05.197305 INFO     ----------------------- Realsense Cameras ----------------------------------                                                                                    image_server.py:785
06:11:05.197626 INFO     RealSense serial numbers: ['348522075219']                                                                                                                      image_server.py:786
06:11:05.197857 INFO     RealSense video paths: ['/dev/video0', '/dev/video1', '/dev/video2', '/dev/video3', '/dev/video4', '/dev/video5']                                               image_server.py:787
06:11:05.198068 INFO     RealSense RGB-like video paths: ['/dev/video2', '/dev/video4']                                                                                                  image_server.py:788
06:11:05.198283 INFO     =========================== Camera Discovery End ================================