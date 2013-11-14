package softcomputing.BrailleConverter;

import static com.googlecode.javacv.cpp.opencv_core.IPL_DEPTH_8U;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateImage;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSize;
import static com.googlecode.javacv.cpp.opencv_highgui.cvLoadImage;
import static com.googlecode.javacv.cpp.opencv_highgui.cvSaveImage;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_RGB2GRAY;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_THRESH_BINARY;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvCvtColor;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvThreshold;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class Program {
	/**
	 * @param args
	 * @throws FileNotFoundException
	 */
	public static void main(String[] args) throws FileNotFoundException {
		System.out.println("Welcome at MLP program");
		try {
			File logsFile = new File("logs.txt");
			FileWriter fw = new FileWriter(logsFile);
			fw.write("Params \n");
			fw.write("L1\t L2\t L3\t TF\t LnR\t XiT\t XeR\t BTCH\t DInS\t DOuS\t WghF\t WghT\n");
			// set up variables
			String imageDirPath = "C:\\Users\\luke\\workspace\\SoftCompProject\\src\\alphabet";				// œcie¿ka przeznaczona na pliki do przetworzenia
			ArrayList<Integer> neuronsInLayers = new ArrayList<Integer>();
			neuronsInLayers.clear();
			neuronsInLayers.add(300);//15*20
			neuronsInLayers.add(15);
			neuronsInLayers.add(6);

			double learningRate = 0.2;
			int maxIteration = 100000;			//
			double maxError = 0.2;				// warunki przerwania uczenia
			boolean batchMode = false;
			int dataSetInputSize = 300;//15*20
			int dataSetOutputSize = 6; // rozpoznajemy 6 liter a,i,m,o,u,x
			
			
			String folderBase = "C:\\Users\\luke\\workspace\\SoftCompProject\\src\\Test\\";					// œcie¿ka bazowa do folderów zawieraj¹cych obiekty treningowe - z podzia³em na klasy (w folderze jest jedna klasa)
			ArrayList<String> classFoldersPathes = new ArrayList<String>();
			classFoldersPathes.clear();
			classFoldersPathes.add(folderBase + "A");		// foldery - z plikami testowymi - klasy
			classFoldersPathes.add(folderBase + "I");
			classFoldersPathes.add(folderBase + "M");
			classFoldersPathes.add(folderBase + "O");
			classFoldersPathes.add(folderBase + "U");
			classFoldersPathes.add(folderBase + "X");
			
			int randomWeightFrom = -1;
			int randomWeightTo = 1;

			/* select the mode of work */
			if (args.length > 0 && args[0].equalsIgnoreCase("Preprocess")) {
				Preprocess(imageDirPath, 0.2);
			} else if (args.length > 0 && args[0].equalsIgnoreCase("Learn")) {
				
				//learn network
				double error = NeuralNetworkLearning(neuronsInLayers,
						TransferFunctionType.TANH, learningRate, maxIteration,
						maxError, batchMode, dataSetInputSize,
						dataSetOutputSize, classFoldersPathes,
						randomWeightFrom, randomWeightTo);
				
				System.out.println("Error computed");
				// log results
				fw.write(neuronsInLayers.get(0) + "\t" + neuronsInLayers.get(1)
						+ "\t" + neuronsInLayers.get(2) + "\t" + "S" + "\t"
						+ learningRate + "\t" + maxIteration + "\t" + maxError
						+ "\t" + batchMode + "\t" + dataSetInputSize + "\t"
						+ dataSetOutputSize + "\t" + randomWeightFrom + "\t"
						+ randomWeightTo + " -------------------------------"
						+ error + "\n");

			} else {
				System.out.println("Wrong parameters");
			}
			fw.write(neuronsInLayers.get(0) + "\t" + neuronsInLayers.get(1)
					+ "\t" + neuronsInLayers.get(2) + "\t" + "S" + "\t"
					+ learningRate + "\t" + maxIteration + "\t" + maxError
					+ "\t" + batchMode + "\t" + dataSetInputSize + "\t"
					+ dataSetOutputSize + "\t" + randomWeightFrom + "\t"
					+ randomWeightTo + "\n");

			System.out.println("Successfully finished");
			fw.write("Finished");
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void Preprocess(String imageDirectoryPath, double noiseRatio) {
		System.out.println("Preprocessing started.");
		/* extract file path from folder */
		ArrayList<ImageFileInfo> files = GetFileNamesList(imageDirectoryPath);

		for (int i = 0; i < files.size(); i++) {
			/* Load image */
			IplImage originalImg = cvLoadImage(files.get(i).get_filePath());

			/* Convert into gray scale */
			IplImage grayImage = IplImage.create(originalImg.cvSize(),
					IPL_DEPTH_8U, 1);
			cvCvtColor(originalImg, grayImage, CV_RGB2GRAY);

			/* Add noise */
			/*
			 * 
			 * To be implemented
			 */
			System.out.println("Noise ratio " + noiseRatio);

			/* Threshold it */
			IplImage imgThreshold = cvCreateImage(cvGetSize(originalImg), 8, 1);
			cvThreshold(grayImage, imgThreshold, 128, 255, CV_THRESH_BINARY);
			
			/* Resize */
			IplImage resizedImg = IplImage.create(15,20, imgThreshold.depth(), imgThreshold.nChannels());
			cvResize(imgThreshold, resizedImg, CV_INTER_LINEAR);
			
			/* Save image into file */
			cvSaveImage(files.get(i).get_directoryPath() + "\\" + files.get(i).get_fileName() + "_thresholded.png", resizedImg);
			System.out.println("Successfully processed image :"
					+ files.get(i).get_fileName());
		}
	}

	public static double NeuralNetworkLearning(
			ArrayList<Integer> neuronsInLayers,
			TransferFunctionType transferFuncType, double learningRate,
			int maxIteration, double maxError, boolean batchMode,
			int dataSetInputSize, int dataSetOutputSize,
			ArrayList<String> classFoldersPathes, int randomWeightFrom,
			int randomWeightTo) {
		System.out.println("Learning option");
		MultiLayerPerceptron mlpNet = null;

		// activation function
		mlpNet = new MultiLayerPerceptron(neuronsInLayers, transferFuncType);

		// Backpropagation configuration
		BackPropagation BPLearningMethod = new BackPropagation();

		// Select training dataset
		DataSet trainingDataset = new DataSet(dataSetInputSize,
				dataSetOutputSize);
		for (int i = 0; i < classFoldersPathes.size(); i++) {
			ArrayList<ImageFileInfo> fileNames = GetFileNamesList(classFoldersPathes
					.get(i));
			System.out.println("file names detected for " + classFoldersPathes.get(i));
			for (int j = 0; j < fileNames.size(); j++) { //fileNames.size()
				IplImage img = cvLoadImage(fileNames.get(j).get_filePath());
				DataSetRow row = GenerateDataSetRowFromImage(img, i,
						dataSetInputSize, classFoldersPathes.size());
				trainingDataset.addRow(row);
				System.out.println("dataset row added : " + j);
			}
			
			System.out.println("folder processed " + classFoldersPathes.get(i));
		}
		// assigning values
		BPLearningMethod.setLearningRate(learningRate);
		BPLearningMethod.setMaxIterations(maxIteration);
		BPLearningMethod.setMaxError(maxError);
		BPLearningMethod.setBatchMode(batchMode);
		BPLearningMethod.setTrainingSet(trainingDataset);
		
		System.out.println("Parameters for Backpropagation set");
		// select initial weights
		mlpNet.randomizeWeights(randomWeightFrom, randomWeightTo);
		// learning method
		System.out.println("Weights randomized");
		mlpNet.learn(trainingDataset, BPLearningMethod);
		System.out.println("MLP learnt");
		/*
		 * Testing Network
		 */

		System.out.println("Learning error = "
				+ BPLearningMethod.getTotalNetworkError());

		// set network input
		// mlpNet.setInput(1, 1, 0);

		// calculate network
		// mlpNet.calculate();

		// get network output
		// double[] networkOutput = mlpNet.getOutput();

		// System.out.println("output : " + networkOutput[0] + " " +
		// networkOutput[1]+ " " + networkOutput[2]);
		// System.out.println("error " +
		// BPLearningMethod.getTotalNetworkError());

		return BPLearningMethod.getTotalNetworkError();
	}

	public static ArrayList<ImageFileInfo> GetFileNamesList(String folder) {

		File directory = new File(folder);
		File[] contents = directory.listFiles();

		ArrayList<ImageFileInfo> filePaths = new ArrayList<ImageFileInfo>();
		for (File f : contents) {
			ImageFileInfo fileInfo = new ImageFileInfo();
			fileInfo.set_fileName(f.getName().substring(0, f.getName().length()-4));
			fileInfo.set_directoryPath(f.getParent());
			fileInfo.set_filePath(f.getAbsolutePath());
			filePaths.add(fileInfo);
		}
		return filePaths;
	}

	public static DataSetRow GenerateDataSetRowFromImage(IplImage image,
			int classNo, int inputSize, int outputSize) {
		double[] input = new double[inputSize];
		
		CvMat imageMat = image.asCvMat();
		int counter = 0;
		System.out.println("DataSetRow");
		for(int i=0; i<imageMat.rows(); i++)
		{
			for(int j=0; j< imageMat.cols(); j++)
			{
				double value8Bit = imageMat.get(i, j);
				
				if(value8Bit == 255)
				{
					input[counter] = 1;
					System.out.print(1);
				}
				else
				{
					input[counter] = 0;
					System.out.print(0);
				}
				counter++;
			}
		}

		double[] desiredOutput = new double[outputSize];
		for (int i = 0; i < outputSize; i++) {
			if (i == classNo) {
				desiredOutput[i] = 1;
			} else {
				desiredOutput[i] = 0;
			}
		}

		DataSetRow row = new DataSetRow(input, desiredOutput);
		return row;
	}
}
