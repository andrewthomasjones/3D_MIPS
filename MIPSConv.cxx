//ITK HEADERS
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkNiftiImageIO.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageFileWriter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhood.h"
#include "itkConvolutionImageFilter.h"
#include "itkConstantBoundaryCondition.h"
#include "itkSubtractImageFilter.h"
#include "itkNeighborhoodOperator.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageToHistogramFilter.h"
#include "itkHistogramThresholdImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkInvertIntensityImageFilter.h"
#include "itkBinaryContourImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "itkPowImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkMRFImageFilter.h"
#include "itkMinimumDecisionRule.h"
#include "itkDistanceToCentroidMembershipFunction.h"
#include "itkManhattanDistanceMetric.h"
#include "itkEuclideanDistanceMetric.h"
#include "itkComposeImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkVectorImageToImageAdaptor.h"
 #include "itkNeighborhoodIterator.h"
//MISC
#include "array"
#include "iostream"
#include "fstream"
#include "cstdio"
#include "cstdlib"
#include "cv.h"       // opencv general include file
#include "ml.h"
#include "string.h"
#include "vector"
#include "unistd.h"
#include "getopt.h"
#include "vector"

#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 

#include <boost/filesystem.hpp>

namespace fs = ::boost::filesystem;

//IMAGE TYPES
typedef float              PixelType;
typedef itk::Image< PixelType, 2 >  ImageType2D;
typedef itk::Image< PixelType, 3 >  ImageType3D;
typedef ImageType3D::Pointer ImagePointer3D;
typedef ImageType2D::Pointer ImagePointer2D;


	
ImagePointer3D NiftiReader(std::string inputFile)
{
   typedef itk::ImageFileReader<ImageType3D> ImageReaderType;
   itk::NiftiImageIO::Pointer niftiIO=itk::NiftiImageIO::New();
   ImageReaderType::Pointer imageReader=ImageReaderType::New();
   ImagePointer3D output;
   try
   {
      niftiIO->SetFileName(inputFile);
      niftiIO->ReadImageInformation();

      imageReader->SetImageIO(niftiIO);
      imageReader->SetFileName(niftiIO->GetFileName());
      output=imageReader->GetOutput();
      output->Update();
      std::cout << "Successfully read: " << inputFile << std::endl;
   }
   catch(itk::ExceptionObject &)
   {
      std::cerr << "Failed to read: " << inputFile << std::endl;
   }
   return output;
}//end of NiftiReader()

bool NiftiWriter(ImagePointer3D input, std::string outputFile)
{
   typedef itk::ImageFileWriter<ImageType3D > imageWriterType;
   imageWriterType::Pointer imageWriterPointer =imageWriterType::New();
   itk::NiftiImageIO::Pointer niftiIO=itk::NiftiImageIO::New();

   try
   {
      //Set the output filename
      imageWriterPointer->SetFileName(outputFile);
      //Set input image to the writer.
      imageWriterPointer->SetInput(input);
      //Determine file type and instantiate appropriate ImageIO class if not explicitly stated with SetImageIO, then write to disk.
      imageWriterPointer->SetImageIO(niftiIO);

      imageWriterPointer->Write();
      std::cout << "Successfully saved image: " << outputFile << std::endl;
      return true;
   }
   catch ( itk::ExceptionObject & ex )
   {
      std::string message;
      message = "Problem found while saving image ";
      message += outputFile;
      message += "\n";
      message += ex.GetLocation();
      message += "\n";
      message += ex.GetDescription();
      std::cerr << message << std::endl;
      return false;
   }
}


// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path& root, const std::string& ext, std::vector<fs::path>& ret)
{
    if(!fs::exists(root) || !fs::is_directory(root)) return;

    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;

    while(it != endit)
    {
        if(fs::is_regular_file(*it) && it->path().extension() == ext) ret.push_back(it->path().filename());
        ++it;

    }

}


ImageType3D::Pointer  slicer(ImageType3D::Pointer inputImage, int rad, int dim){
	
	ImageType3D::RegionType region;
    ImageType3D::RegionType::SizeType size;
    ImageType3D::RegionType::IndexType index;
    ImageType3D::Pointer outputImage = ImageType3D::New();
    ImageType3D::RegionType requestedRegion = inputImage->GetRequestedRegion();
    
    region.SetSize( requestedRegion.GetSize() );
    region.SetIndex( requestedRegion.GetIndex() );
   
    outputImage->SetRegions( region );
    outputImage->Allocate();
    
    itk::NeighborhoodIterator<ImageType3D>::RadiusType radius;
    radius.Fill(rad);
    
    itk::NeighborhoodIterator<ImageType3D> itIn(radius, inputImage, requestedRegion);
    itk::NeighborhoodIterator<ImageType3D> itOut(radius, outputImage, requestedRegion);  

    int c = (itIn.Size());
	int mid = c / 2;
	int len = 2*rad+1;
	
	int step = (int) std::pow(len, dim);
	double setForMax[len]; 
	
	//std::cout << "len " << len << " step " << step  << " c " << c << std::endl;
    while( !itIn.IsAtEnd() )
    {
		
		for(int i=0; i<len; i++){
			setForMax[i]=itIn.GetPixel(mid + (i-rad)*step);
			
		}
		double maxi = *std::max_element(setForMax, setForMax+len);
		
		//std::cout << "   " << maxi << std::endl;
		itOut.SetCenterPixel(maxi);
		
		
		++itIn;
		++itOut;
      }
      
	return outputImage;
}
	
int main(int argc, char *argv[])
{
	
	std::string inFolder(argv[1]);
	std::string outFolder(argv[2]);
	int rad = atoi(argv[3]);
	int dim = atoi(argv[4]);
	const std::string ext = ".nii";
	
	std::cout<< inFolder  << std::endl;
	std::cout<<	outFolder << std::endl;
	
	fs::path inPath(inFolder);
	
	std::vector<fs::path> files;
	
	
	std::cout << "Read file names..." << std::endl;
	get_all(inPath, ext, files);
	
	for (const auto &piece : files){
	//fs::path piece = files.at(6);			
		 
		 std::cout << "Reading ..." + inFolder + "/" + piece.string() << std::endl;
		 ImagePointer3D in = NiftiReader(inFolder+"/"+piece.string());
		 std::cout << "MIPS..."  << std::endl;
		 
		 //int rad =2;
	      //dim =2; // 0,1 or 2
		 ImagePointer3D out = slicer(in,rad,dim); 
		 
		 NiftiWriter(out, outFolder+"/mips_"+std::to_string(rad)+"_"+std::to_string(dim)+"_"+piece.string());
	}
	std::cout << "Done."  << std::endl;
	return 0;
}

//./build/MIPSConv /data/nif02/uqajon14/20150628_5dpf_H2BS_CON_LR_F03/03-chunk  /data/nif02/uqajon14/20150628_5dpf_H2BS_CON_LR_F03/mips_test 2 2

