- The purpose of this project is to deploy and test a CNN model trained on the cifar-10 dataset. This was created for the purpose of testing for the QE team.

- The trained model i.e. state dictionary is at '.model_cifar.pt' 

- The directory sample-cifar-10 contains some images from the cifar-10 dataset 

- In order to use the launcher either upload an image or enter a path of local DFS file

- Only a 32x32 RGB image can be processed.

- The image sent will be classified as one of the following :-

      ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

      example of a successful request:
      	{
		    "message": "success!",
		    "prediction": {
		        "airplane": "0.00032329743",
		        "automobile": "0.012829902",
		        "bird": "0.00096437446",
		        "cat": "9.434413e-05",
		        "deer": "4.26209e-07",
		        "dog": "5.861184e-07",
		        "frog": "3.7358205e-07",
		        "horse": "3.733835e-06",
		        "predicted_class": "truck",
		        "prob_predicted_class": "0.9853125",
		        "ship": "0.00047049456",
		        "truck": "0.9853125"
		    }
		}

      example of a failed request:
		{
		    "error": "only RGB image 32x32 accepted",
		    "message": "failure!"
		}

 - Please email: abishek.subramanian@dominodatalab.com for any questions or concerns 