package Readers

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
)

const (
	trainingImagesPath = ""
	trainingLabelsPath = ""
	testImagesPath     = ""
	testLabelPath
)

// ReadImages reads both training and test images from MNIST dataset
func ReadImages() (trainingImages [][]float64, testImages [][]float64, err error) {
	trainingImages, err = readMnistImages(trainingImagesPath, 60000)
	if err != nil {
		return nil, nil, err
	}

	testImages, err = readMnistImages(testImagesPath, 10000)
	if err != nil {
		return nil, nil, err
	}
	return trainingImages, testImages, nil
}

// ReadLabels reads both training and test labels from MNIST dataset
func ReadLabels() (trainingLabels [][]float64, testLabels [][]float64, err error) {
	trainingLabels, err = readMnistLabels(trainingLabelsPath, 60000)
	if err != nil {
		return nil, nil, err
	}

	testLabels, err = readMnistLabels(testLabelPath, 10000)
	if err != nil {
		return nil, nil, err
	}
	return trainingLabels, testLabels, nil
}

// generic function to read image data from a file
func readMnistImages(filePath string, expectedImages uint32) ([][]float64, error) {
	// open the file
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatal(err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			log.Fatalf("Error closing file: %s", err)
		}
	}(file)

	// read the header
	header := make([]byte, 16)
	_, err = file.Read(header)
	if err != nil {
		log.Fatal(err)
	}

	// parse header information
	magicNumber := binary.BigEndian.Uint32(header[:4])
	numImages := binary.BigEndian.Uint32(header[4:8])
	numRows := binary.BigEndian.Uint32(header[8:12])
	numCols := binary.BigEndian.Uint32(header[12:16])

	log.Println("image file:", filePath)
	log.Printf("magic number: %d, number of images: %d, number of rows: %d, number of columns: %d", magicNumber, numImages, numRows, numCols)

	// validate header
	if magicNumber != 2051 {
		return nil, fmt.Errorf("invalid magic number: %d", magicNumber)
	}

	if numImages != expectedImages {
		return nil, fmt.Errorf("expected %d images, got %d", expectedImages, numImages)
	}

	// read images data
	images := make([][]float64, numImages)
	for i := range images {
		images[i] = make([]float64, numRows*numCols)
		template := make([]byte, numRows*numCols)
		_, err = file.Read(template)
		if err != nil {
			log.Fatal(err)
		}
		// normalize pixel values to [0,1]
		for j := range template {
			images[i][j] = float64(template[j]) / 255.0
		}
	}

	return images, nil
}

// generic function to read label data from a file
func readMnistLabels(filePath string, expectedLabels uint32) ([][]float64, error) {
	// open file
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatal(err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			log.Fatalf("Error closing file: %s", err)
		}
	}(file)

	// read header
	header := make([]byte, 8)
	_, err = file.Read(header)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}

	// parse header information
	magicNumber := binary.BigEndian.Uint32(header[:4])
	numLabels := binary.BigEndian.Uint32(header[4:8])

	log.Println("image file:", filePath)
	log.Printf("magic number: %d, number of labels: %d", magicNumber, numLabels)

	// validate header
	if magicNumber != 2049 {
		return nil, fmt.Errorf("invalid magic number: %d", magicNumber)
	}

	if numLabels != expectedLabels {
		return nil, fmt.Errorf("expected %d labels, got %d", expectedLabels, numLabels)
	}

	// read label data
	labels := make([][]float64, numLabels)
	for i := range labels {
		// one-hot encoding for digit 0-9
		labels[i] = make([]float64, 10)

		var label uint8
		err := binary.Read(file, binary.BigEndian, &label)
		if err != nil {
			log.Fatal(err)
		}

		labels[i][label] = 1.0
	}

	return labels, nil

}

// ReadData is the original function for backward compatibility
// it read both images and labels from MNIST dataset
func ReadData() (trainingImages [][]float64, trainingLabels [][]float64, testImages [][]float64, testLabels [][]float64, err error) {
	trainingImages, testImages, err = ReadImages()
	if err != nil {
		return nil, nil, nil, nil, err
	}

	testImages, testLabels, err = ReadLabels()
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return trainingImages, trainingLabels, testImages, testLabels, nil
}
