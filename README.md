# gtaVision
Computer vision for GTA V self-driving car

# Demo
Here is a demo of what the end result will look like for computer vision in GTA V. 
![after recognition](https://github.com/NoahNaugle/gtaVision/blob/master/Img/computer_vision_demo_recognition.png)

# VOC dataset
I am in the process of creating the VOCGTA dataset. It should be around ~20,000 images with ~25 categories. The VOC dataset will follow the exact same format the native VOC datasets are to make it easy to work with/train. The goal for the object recognition portion is to aid the AI to make decisions. Currently the AI is composed of a CNN that takes images and key presses for each frame. Sentdex the original creator of the self-driving car in GTA V has been working on improving the confidence level and sensitivity of the AI's driving movements. This repository will aim at strengthening the vision to be not only 

```
frame + key_press + minimap = 3 
BUT
frame + key_press + minimap+ 25(object_recognition) = 28 
```
## AI situation reactions
Enabling the object recognition aid will allow the AI to make better decisions depending on what it sees. 
here are a few examples of situations the AI may encounter.

**1. AI sees a civilian sign**
```
*slows down 5 mph
```
**2. AI see a stop sign**
```
*slows down to stop until straight line on the road goes away.
(GTA has a horizontal line at almost every stop sign on the road, and at the angle of the
hood camera could allow the car to slow down till it could not see it, kinda hard to explain)
```
**3. AI sees red, yellow, or green light**
```
RED *stops at horizontal line on road before intersection, or behind car.
YELLOW *slows to a stop a horizontal line in road or continues through light depending on 
  distance (distance measuring will be a challenge, but I think some people have attepted it in some 
  papers I have read.)
GREEN *proceeds through light.
```
**4. AI sees yield sign**
```
*slows down to horizontal line on road and waits 3 seconds then goes on path.
```

**5. AI sees NoUTurn, Turn Left, or Right sign**
```NoUTurn *AI cannot turn around at that opening in the road.
Turn Left *AI can turn left in the indicated or current lane.
Turn Right *AI can turn right in the indicated lane or current lane.

I say "indicated lane" because most of the roads have road markings like turn left painted on the road, 
like you normally would see in the real life. Indicated that the lane has a turn marking on the road enables 
the AI to make the move, else straight.
```
**6. AI sees speed limit sign**
```
*changes current speed to speed of speed limit +-5 while flowing with traffic, 
and traffic being the priority over speed.
```
**7. AI sees road construction sign**
```
*slows down 10mph and navigates through road construction lane change/closed off.
```

There a far many more situations that the AI will encounter that I have not mentioned. Some of the decision making will be purely training based. For example, how to drive/maintain lanes. It will take 100's of hours of training and balancing the data before the AI can confidently make decisions on the road. As for the object recognition, it serves as a backbone for the AI. As you can imagine, if you were the AI you would have no clue what to do. The GTA world is so complex, it needs to be specifically trained to recognize what is coming around the corner, and to be prepared for the unexpected. 

![after recognition](https://github.com/NoahNaugle/gtaVision/blob/master/Img/CNN_SSD_training_ELI5.jpg)
Above is some proven tests on the data/images I gather using tensorflow. 

## Other considerations
After the object recognition is completed, trained and passed complete stage, I would be very interested in training agents like a firefighter, taxi driver that reads directions from the minimap, or a semi-truck driver. Each has to respond to different situations differently, which could be cool to watch how each interacts with the environment. This is very far down the road in the project I just thought I would mention it.




