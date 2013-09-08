/* 
* Molecular Visualization HTTP Server
* Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* aint with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
* Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
*
*/

#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#define DEFAULT_LIGHT_MATERIAL 1029

#include <lacewing.h>

#include <map>
#include <vector>
#include <time.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <fstream>
#include <algorithm> 

#include <Consts.h>
#include <PDBReader.h>
#include <FileMarshaller.h>
#include <Logging.h>

#ifdef USE_CUDA
#include <Cuda/CudaKernel.h>
#else
#include <CPUKernel.h>
#endif // USE_CUDA

#include "wininet.h" // for clearing URL cache DeleteUrlCacheEntry
#pragma comment(lib, "wininet.lib") // for clearing URL cache DeleteUrlCacheEntry

extern bool jo_write_jpg(const char *filename, const void *data, int width, int height, int comp, int quality);

const int NB_MAX_SERIES = 5;

// Structures
struct MoleculeInfo
{
   std::string moleculeId;
   int structureType;
   int scheme;
   float3 viewPos;
   float3 rotationAngles;
   SceneInfo sceneInfo;
   PostProcessingInfo postProcessingInfo;
};

struct ChartInfo
{
   int chartType;
   std::vector<float> values[NB_MAX_SERIES];
   float3 viewPos;
   float3 rotationAngles;
   SceneInfo sceneInfo;
   PostProcessingInfo postProcessingInfo;
};

struct IrtInfo
{
   std::string filename;
   float3 viewPos;
   float3 rotationAngles;
   SceneInfo sceneInfo;
   PostProcessingInfo postProcessingInfo;
};

// Requests
std::map<std::string,std::string> gRequests;

// ----------------------------------------------------------------------
// Stats
// ----------------------------------------------------------------------
int gNbCalls(0);

// ----------------------------------------------------------------------
// Usecases
// ----------------------------------------------------------------------
enum UseCase 
{
   ucUndefined = 0,
   ucChart = 1,
   ucIRT   = 2,
   ucPDB   = 3
};
UseCase gCurrentUsecase(ucUndefined);
std::string gCurrentUsecaseValue("undefined");

// ----------------------------------------------------------------------
// Charts
// ----------------------------------------------------------------------
int gChartStartIndex=0;

// ----------------------------------------------------------------------
// Molecules
// ----------------------------------------------------------------------
size_t gCurrentProtein(0);
std::vector<std::string> gProteinNames;

// ----------------------------------------------------------------------
// Scene
// ----------------------------------------------------------------------
#ifdef USE_CUDA
CudaKernel* gpuKernel = nullptr;
#else
CPUKernel* gpuKernel = nullptr;
#endif // USE_CUDA

unsigned int gWindowWidth  = 4096;
unsigned int gWindowHeight = 4096;
unsigned int gWindowDepth  = 4;

float4 gBkGrey  = {0.5f, 0.5f, 0.5f, 0.f};
float4 gBkBlack = {0.f, 0.f, 0.f, 0.f};
int   gTotalPathTracingIterations = 100;
int4  gMisc = {otJPEG,0,0,2};

SceneInfo gSceneInfo;

bool   gSceneHasChanged(true);
bool   gSpecular(true);
bool   gAnimate(false);
int    gTickCount(0);
float  gDefaultAtomSize(100.f);
float  gDefaultStickSize(80.f);
int    gMaxPathTracingIterations = gTotalPathTracingIterations;
int    gNbMaxBoxes( 8*8*8 );
int    gGeometryType(0);
int    gAtomMaterialType(0);
int    gBox(0);
float3 gRotationCenter = { 0.f, 0.f, 0.f };

// Scene description and behavior
int gNbBoxes      = 0;
int gNbPrimitives = 0;
int gNbLamps      = 0;
int gNbMaterials  = 0;

// Camera information
float3 gViewPos    = { 0.f, 0.f, -5000.f };
float3 gViewDir    = { 0.f, 0.f, -2000.f };
float3 gViewAngles = { 0.f, 0.f, 0.f };

// ----------------------------------------------------------------------
// Post processing
// ----------------------------------------------------------------------
PostProcessingInfo gPostProcessingInfo;

// ----------------------------------------------------------------------
// Utils
// ----------------------------------------------------------------------
void saturatefloat4(float4& value, const float min, const float max )
{
   value.x = (value.x < min) ? min : value.x;
   value.y = (value.y < min) ? min : value.y;
   value.z = (value.z < min) ? min : value.z;
   value.x = (value.x > max) ? max : value.x;
   value.y = (value.y > max) ? max : value.y;
   value.z = (value.z > max) ? max : value.z;
}

void readfloats(const std::string value, std::vector<float>& values )
{
   std::string element;
   int i(0);
   for( int j(0); j<value.length(); ++j)
   {
      if( value[j] == ',' )
      {
         values.push_back(static_cast<float>(atof(element.c_str())));
         element = "";
         i++;
      }
      else
      {
         element += value[j];
      }
   }
   if( element.length() != 0 )
   {
      values.push_back(static_cast<float>(atof(element.c_str())));
   }
}

float3 readfloat3(const std::string value)
{
   float3 result = {0.f,0.f,0.f};
   std::string element;
   int i(0);
   for( int j(0); j<value.length(); ++j)
   {
      if( value[j] == ',' )
      {
         switch( i )
         {
         case 0: result.x = static_cast<float>(atof(element.c_str())); break;
         case 1: result.y = static_cast<float>(atof(element.c_str())); break;
         case 2: result.z = static_cast<float>(atof(element.c_str())); break;
         }
         element = "";
         i++;
      }
      else
      {
         element += value[j];
      }
   }
   if( element.length() != 0 )
   {
      result.z = static_cast<float>(atof(element.c_str()));
   }
   return result;
}

/*
________________________________________________________________________________

Create Random Materials
________________________________________________________________________________
*/
void createMaterials( GPUKernel* gpuKernel, const bool& random )
{
   int start(0);
   int end(NB_MAX_MATERIALS);
	// Materials
	for( int i(start); i<end; ++i ) 
	{
		float4 specular = {0.f,0.f,0.f,0.f};
		specular.x = 0.1f;
		specular.y = 100.f;
		specular.z = 0.f;
		specular.w = 0.f;

		float reflection   = 0.f;
		float refraction   = 0.f;
		float transparency = 0.f;
		int   textureId = TEXTURE_NONE;
      float3 innerIllumination = { 0.f, 40000.f, gSceneInfo.viewDistance.x };
		bool procedural = false;
		bool wireframe = false;
		int  wireframeDepth = 0;
		float r,g,b,noise;
      bool fastTransparency = false;
      float fastTransparencyRefraction = 0.95f;
       
		r = rand()%1000/1000.f;
		g = rand()%1000/1000.f;
		b = rand()%1000/1000.f;
      noise = 0.f;

		switch( i )
		{
      // Molecules
		case 0: r = 0.8f;        g = 0.7f;        b = 0.7f;         break; 
		case 1: r = 0.7f;        g = 0.7f;        b = 0.7f;         break; // C Gray
		case 2: r = 174.f/255.f; g = 174.f/255.f; b = 233.f/255.f;  break; // N Blue
		case 3: r = 0.9f;        g = 0.4f;        b = 0.4f;         break; // O 
		case 4: r = 0.9f;        g = 0.9f;        b = 0.9f;         break; // H White
		case 5: r = 0.0f;        g = 0.5f;        b = 0.6f;         break; // B
		case 6: r = 0.0f;        g = 0.0f;        b = 0.7f;         break; // F
		case 7: r = 0.8f;        g = 0.6f;        b = 0.3f;         break; // P
		case 8: r = 241.f/255.f; g = 196.f/255.f; b = 107.f/255.f;  break; // S Yellow
		case 9: r = 0.9f;        g = 0.3f;        b = 0.3f;         break; // V

      // transparent Molecules
      case 10: r = 0.f; g = 0.f; b = 1.f; fastTransparency=true; transparency=0.9f; refraction=fastTransparencyRefraction; specular.x=0.1f; specular.y=100.f; break; 
      case 11: r = 1.f; g = 1.f; b = 1.f; specular.x=0.1f; specular.y=10.f; break; 

      case 100: r=1.f;  g=1.f;  b=1.f; noise=0.f; specular.x=0.1f; specular.y=10.f; reflection=0.3f; break;

      // Sky Box  
		case 101: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 0; break; 
		case 102: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 1; break; 
		case 103: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 2; break; 
		case 104: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 3; break; 
		case 105: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 4; break; 
		case 106: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 5; break; 
      
      // Cornell Box
      case 107: r=127.f/255.f; g=127.f/255.f; b=127.f/255.f; specular.x = 0.2f; specular.y = 10.f;  specular.w = 0.3f; break;
      case 108: r=154.f/255.f; g=94.f/255.f;  b=64.f/255.f;  specular.x = 0.1f; specular.y = 100.f; specular.w = 0.1f; break;
		case 109: r=92.f/255.f;  g=93.f/255.f;  b=150.f/255.f; specular.x = 0.3f; specular.y = 20.f;  specular.w = 0.5f; break;
		case 110: r=92.f/255.f;  g=150.f/255.f; b=93.f/255.f;  specular.x = 0.3f; specular.y = 20.f;  specular.w = 0.5f; break;
		
      // Fractals
      case 111: r=127.f/255.f; g=127.f/255.f; b=127.f/255.f; specular.x = 0.2f; specular.y = 10.f;  specular.w = 0.3f; wireframe=false; textureId=TEXTURE_MANDELBROT; break;
      case 112: r=154.f/255.f; g=94.f/255.f;  b=64.f/255.f;  specular.x = 0.1f; specular.y = 100.f; specular.w = 0.1f; wireframe=false; textureId=TEXTURE_JULIA; break;

      // Basic reflection
      case 113: /*r=0.5f; g=1.0f; b=0.7f; */reflection = 0.5f; refraction=1.6f; transparency=0.7f; break;
		case 114: /*r=1.f; g=1.f; b=1.f;*/ reflection = 0.9f; break;
      case 115: r=0.5f; g=1.0f; b=0.7f; reflection = 0.f; textureId = 0; break;
      case 116: /*r=0.f; g=0.f; b=0.f;*/ reflection = 0.1f; refraction=1.66f; transparency=0.5f; specular.x = 0.5f; specular.y = 10.f;break;
		case 117: r=1.f; g=0.f; b=0.f; reflection = 0.5f; break;
		case 118: r=0.f; g=1.f; b=1.f; reflection = 0.5f; break;

      // White
      case 119: r=1.f; g=1.f; b=1.f; break;

      // Wireframe
         /*
		case 120: r=1.f; g=0.f; b=0.f; wireframe = true; break;
		case 121: r=0.f; g=1.f; b=0.f; wireframe = true; break;
		case 122: r=0.f; g=0.f; b=1.f; wireframe = true; break;
      */
      case 120: innerIllumination.x=.5f; break; 
      case 121: innerIllumination.x=.5f; break; 
      case 122: innerIllumination.x=.5f; break; 
      case 123: innerIllumination.x=.5f; break; 
		case 124: innerIllumination.x=.5f; break; 
		case 125: innerIllumination.x=.5f; break; 
		case 126: innerIllumination.x=.5f; break; 
		case 127: innerIllumination.x=.5f; break; 
		case 128: innerIllumination.x=.5f; break; 
		case DEFAULT_LIGHT_MATERIAL: r=1.f; g=1.f; b=1.f; innerIllumination.x=1.f; break; 

		default:
         if( i<60 )
         {
            if( i%2==0 ) 
            {
               //fastTransparency = true;
               transparency=0.7f; 
               refraction=fastTransparencyRefraction;
            }

			   r = 0.2f+rand()%800/1000.f;
			   g = 0.2f+rand()%800/1000.f;
			   b = 0.2f+rand()%800/1000.f;
            switch( rand()%8 )
            {
            case 0: 
               {
                  //textureId=rand()%10; 
                  break;
               }
            case 1:
               {
				      reflection = 0.9f;
			         specular.x = 1.0f;
			         specular.y = 10.f*(1+rand()%50);
		            specular.w = 0.0f;
                  break;
               }
            case 2:
               {
				      refraction = 0.8f+rand()%2000/1000.f; 
				      transparency= 0.6f+rand()%400/1000.f;
				      reflection = rand()%1000/1000.f;
                  break;
               }
            case 3:
               {
				      refraction = 0.8f+rand()%2000/1000.f; 
				      transparency= 0.6f+rand()%400/1000.f;
				      reflection = rand()%1000/1000.f;
                  break;
               }
            case 4:
               {
				      reflection = rand()%1000/1000.f;
                  break;
               }
            }
			   break;
		   }
         else
         {
            // from index 60, materials are textured!
            if( (i-60)<static_cast<int>(gpuKernel->getNbActiveTextures()) )
            {
               textureId = i-60;
            }
            else
            {
               reflection = 1.f;
            }
         }
      }
      int material = gpuKernel->addMaterial();
		gpuKernel->setMaterial(
			material, r, g, b, noise,
			reflection, refraction, procedural, 
			wireframe, wireframeDepth,
			transparency, textureId,
			specular.x, specular.y, specular.w, 
         innerIllumination.x, innerIllumination.y, innerIllumination.z,
			fastTransparency);
	}

   gpuKernel->compactBoxes(false);
}

void initializeKernel( const bool& random )
{
   gpuKernel->resetAll();
   gpuKernel->setFrame(0);

   createMaterials( gpuKernel, random );

   /*
	// Textures
   HANDLE hFind;
   WIN32_FIND_DATA FindData;

   std::string path("./textures/");
   std::string fullFilter(path);
   fullFilter += "*.bmp";
   hFind = FindFirstFile(fullFilter.c_str(), &FindData);
   int i(0);
   while (FindNextFile(hFind, &FindData))
   {
      std::string fullPath(path);
      fullPath+=FindData.cFileName;
      int slot = gpuKernel->loadTextureFromFile(i,fullPath);
      LOG_INFO(3, "Texture " << fullPath << " loaded into slot " << slot );
      i++;
   }
   */
}

void destroyKernel()
{
   //delete gpuKernel;
   //gpuKernel = nullptr;
}

void initializeMolecules()
{
   // Proteins vector
   gProteinNames.push_back("3VM9");
   gProteinNames.push_back("1BNA");
   gProteinNames.push_back("3SUI");
   gProteinNames.push_back("1ACY");
   gProteinNames.push_back("3VHS");
   gProteinNames.push_back("4FMC");
   gProteinNames.push_back("3TGW");
   gProteinNames.push_back("4FI3");
   gProteinNames.push_back("3VJM");
   gProteinNames.push_back("4FME");
   gProteinNames.push_back("3U7D");
   gProteinNames.push_back("3U2Z");
   gProteinNames.push_back("3UA5");
   gProteinNames.push_back("3VKL");
   gProteinNames.push_back("3VKM");
}

static char encoding_table[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
   'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
   'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
   'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
   'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
   'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
   'w', 'x', 'y', 'z', '0', '1', '2', '3',
   '4', '5', '6', '7', '8', '9', '+', '/'};
static char *decoding_table = nullptr;
static int mod_table[] = {0, 2, 1};

char *base64_encode(const unsigned char *data,
   size_t input_length,
   size_t *output_length) 
{
   *output_length = (size_t) (4.0 * ceil((double) input_length / 3.0));

   char *encoded_data = (char*)malloc(*output_length+1);
   if (encoded_data == NULL) return NULL;

   for (int i = 0, j = 0; i < input_length;) {

      uint32_t octet_a = i < input_length ? data[i++] : 0;
      uint32_t octet_b = i < input_length ? data[i++] : 0;
      uint32_t octet_c = i < input_length ? data[i++] : 0;

      uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

      encoded_data[j++] = encoding_table[(triple >> 3 * 6) & 0x3F];
      encoded_data[j++] = encoding_table[(triple >> 2 * 6) & 0x3F];
      encoded_data[j++] = encoding_table[(triple >> 1 * 6) & 0x3F];
      encoded_data[j++] = encoding_table[(triple >> 0 * 6) & 0x3F];
   }

   for (int i = 0; i < mod_table[input_length % 3]; i++)
   {
      encoded_data[*output_length - 1 - i] = '=';
   }

   encoded_data[*output_length] = 0;
   return encoded_data;
}

char* convertToBMP( char* buffer )
{
   unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,  0,0};
   unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0,  1,0, 24,0};
   unsigned char bmppad[3] = {0,0,0};

   int w = gWindowWidth;
   int h = gWindowHeight;
   int filesize = 54 + gWindowDepth*w*h;

   bmpfileheader[ 2] = (unsigned char)(filesize    );
   bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
   bmpfileheader[ 4] = (unsigned char)(filesize>>16);
   bmpfileheader[ 5] = (unsigned char)(filesize>>24);

   bmpinfoheader[ 4] = (unsigned char)(       w    );
   bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
   bmpinfoheader[ 6] = (unsigned char)(       w>>16);
   bmpinfoheader[ 7] = (unsigned char)(       w>>24);

   bmpinfoheader[ 8] = (unsigned char)(       h    );
   bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
   bmpinfoheader[10] = (unsigned char)(       h>>16);
   bmpinfoheader[11] = (unsigned char)(       h>>24);

   char* result = new char[filesize];
   memcpy(result   ,bmpfileheader,14);
   memcpy(result+14,bmpinfoheader,40);
   memcpy(result+54,buffer,gWindowDepth*w*h);
   return result;
}

void saveToJPeg( Lacewing::Webserver::Request& request, const std::string& filename, const SceneInfo& sceneInfo, const unsigned char* image )
{
   size_t len(0);
   char* buffer = nullptr;
   long bufferLength;
   jo_write_jpg(filename.c_str(), image, sceneInfo.width.x, sceneInfo.height.x, 3, 100 );
   FILE * pFile;
   size_t result;

   pFile = fopen ( filename.c_str(), "rb" );
   if (pFile!=NULL) 
   {
      // obtain file size:
      fseek (pFile , 0 , SEEK_END);
      bufferLength = ftell (pFile);
      rewind (pFile);

      // allocate memory to contain the whole file:
      buffer = new char[bufferLength];

      // copy the file into the buffer:
      result = fread (buffer,1,bufferLength,pFile);
      if (result != bufferLength) 
      {
         //fputs ("Reading error",stderr); exit (3);
      }

      /* the whole file is now loaded in the memory buffer. */

      // terminate
      fclose (pFile);

      request << "data:image/jpg;base64,";
      request << base64_encode( (const unsigned char*)buffer, bufferLength, &len );
      request.AddHeader("Access-Control-Allow-Origin", "*"); // Needed by Chrome!!
      delete [] buffer;
   }
}

void buildAreaChart( Lacewing::Webserver::Request& request, ChartInfo& chartInfo, const bool& update )
{
   int frame(0);
   float3 cameraOrigin = chartInfo.viewPos;
   float3 cameraTarget = chartInfo.viewPos;
   cameraTarget.z += 10000.f;
   float3 cameraAngles = gViewAngles;

   float3 columnSize    = { 400.f, 40.f, 400.f };
   float3 columnSpacing = { 400.f, 40.f, 800.f };
   float3 size = {500.f,500.f,500.f};
   int material = 0;

   SceneInfo sceneInfo = chartInfo.sceneInfo;
   size_t len(sceneInfo.width.x*sceneInfo.height.x*gWindowDepth);
   long renderingTime = GetTickCount();
   int index(0);

   // Ground
   float sideSize = columnSpacing.x*chartInfo.values[0].size()*0.9f;
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
   if(!update) gChartStartIndex = gNbPrimitives;
   gpuKernel->setPrimitive( gNbPrimitives, 
         -sideSize, -10.f, -sideSize, 
         sideSize, -10.f, -sideSize,
         sideSize, -10.f,  sideSize,
               0.f, -10.f,       0.f, 
         material);

   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
   gpuKernel->setPrimitive( gNbPrimitives, 
         sideSize, -10.f,  sideSize, 
         -sideSize, -10.f,  sideSize,
         -sideSize, -10.f, -sideSize,
               0.f, 0.f,       0.f, 
         material);

   // Wall
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
   gpuKernel->setPrimitive( gNbPrimitives, 
         -sideSize,         -10.f,  sideSize, 
         sideSize,         -10.f,  sideSize,
         sideSize, sideSize-10.f,  sideSize,
               0.f,      0.f,       0.f, 
         material);
      
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
   gpuKernel->setPrimitive( gNbPrimitives, 
         sideSize, sideSize-10.f,  sideSize, 
         -sideSize, sideSize-10.f,  sideSize,
         -sideSize,         -10.f,  sideSize,
               0.f,      0.f,       0.f, 
         material);

   // Right Side
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
   gpuKernel->setPrimitive( gNbPrimitives, 
         sideSize,         -10.f, -sideSize, 
         sideSize,         -10.f,  sideSize+10.f,
         sideSize, sideSize-10.f,  sideSize+10.f,
               0.f,      0.f,       0.f, 
         material);

   // Left Side
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
   gpuKernel->setPrimitive( gNbPrimitives, 
         -sideSize,         -10.f, -sideSize, 
         -sideSize,         -10.f,  sideSize+10.f,
         -sideSize, sideSize-10.f,  sideSize+10.f,
               0.f,      0.f,       0.f, 
         material);

   // Lamp
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptXZPlane ) : gChartStartIndex+(index++);
   gpuKernel->setPrimitive( gNbPrimitives,  static_cast<float>(rand()%10000-5000), 5000.f, -2000.f-static_cast<float>(rand()%5000), 2000.f, 0.f, 500.f, DEFAULT_LIGHT_MATERIAL);

   // Build Chart
   for( int s(0); s<NB_MAX_SERIES; ++s )
   {
      material = 20+s*5;
      float x=-(columnSpacing.x*chartInfo.values[s].size())/2.f + columnSpacing.x/4.f;

      std::vector<float>::const_iterator it = chartInfo.values[s].begin();
      float value=(*it);
      ++it;
      for( int i(0); i<chartInfo.values[s].size()-1; ++i )
      {
         float ymin = (value< (*it)) ? value : (*it);

         float offsetZ = s*columnSpacing.z - ( NB_MAX_SERIES * columnSpacing.z )/2.f;
            
         // Front
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
         gpuKernel->setPrimitive( gNbPrimitives, 
                           x,                0.f, offsetZ, 
            x+columnSize.x,                0.f, offsetZ,
            x+columnSize.x, ymin*columnSize.y, offsetZ,
                        0.f,                0.f, 0.f, 
                  material);
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
         gpuKernel->setPrimitive( gNbPrimitives, 
            x+columnSize.x, ymin*columnSize.y, offsetZ,
                           x, ymin*columnSize.y, offsetZ,
                           x,                0.f, offsetZ, 
                        0.f,                0.f, 0.f, 
                  material);
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
         if( value < (*it ) )
         {
            gpuKernel->setPrimitive( gNbPrimitives, 
                              x, value*columnSize.y, offsetZ, 
               x+columnSize.x, value*columnSize.y, offsetZ,
               x+columnSize.x, (*it)*columnSize.y, offsetZ,
                           0.f,                0.f, 0.f, 
                     material);
         }
         else
         {
            gpuKernel->setPrimitive( gNbPrimitives, 
                              x, value*columnSize.y, offsetZ, 
                              x, (*it)*columnSize.y, offsetZ,
               x+columnSize.x, (*it)*columnSize.y, offsetZ,
                           0.f,                0.f, 0.f, 
                     material);
         }

         // Back
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
         gpuKernel->setPrimitive( gNbPrimitives, 
                           x,                0.f, columnSize.z + offsetZ, 
            x+columnSize.x,                0.f, columnSize.z + offsetZ,
            x+columnSize.x,  ymin*columnSize.y, columnSize.z + offsetZ,
                        0.f,                0.f, 0.f, 
                  material);
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
         gpuKernel->setPrimitive( gNbPrimitives, 
            x+columnSize.x, ymin*columnSize.y, columnSize.z + offsetZ,
                           x, ymin*columnSize.y, columnSize.z + offsetZ,
                           x,               0.f, columnSize.z + offsetZ, 
                        0.f,               0.f, 0.f, 
                  material);
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
         if( value < (*it ) )
         {
            gpuKernel->setPrimitive( gNbPrimitives, 
                              x, value*columnSize.y, columnSize.z + offsetZ, 
               x+columnSize.x, value*columnSize.y, columnSize.z + offsetZ,
               x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
                           0.f,                0.f, 0.f, 
                     material);
         }
         else
         {
            gpuKernel->setPrimitive( gNbPrimitives, 
                              x, value*columnSize.y, columnSize.z + offsetZ, 
                              x, (*it)*columnSize.y, columnSize.z + offsetZ,
               x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
                           0.f,                0.f, 0.f, 
                     material);
         }

         //top
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
         gpuKernel->setPrimitive( gNbPrimitives, 
                           x, value*columnSize.y, offsetZ, 
            x+columnSize.x, (*it)*columnSize.y, offsetZ,
            x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
                        0.f,                0.f, 0.f, 
                  material);
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
         gpuKernel->setPrimitive( gNbPrimitives, 
            x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
                           x, value*columnSize.y, columnSize.z + offsetZ,
                           x, value*columnSize.y, offsetZ, 
                        0.f,                0.f, 0.f, 
                  material);

         // Sides
         if( i==0 )
         {
            gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
            gpuKernel->setPrimitive( gNbPrimitives, 
                              x,                0.f, offsetZ, 
                              x, value*columnSize.y, offsetZ,
                              x, value*columnSize.y, columnSize.z + offsetZ,
                           0.f,                0.f, 0.f, 
                     material);
            gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
            gpuKernel->setPrimitive( gNbPrimitives, 
                              x, value*columnSize.y, columnSize.z + offsetZ, 
                              x,                0.f, columnSize.z + offsetZ,
                              x,                0.f, offsetZ,
                           0.f,                0.f, 0.f, 
                     material);
         }

         if( i==chartInfo.values[s].size()-2 )
         {
            gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
            gpuKernel->setPrimitive( gNbPrimitives, 
               x+columnSize.x,                0.f, offsetZ, 
               x+columnSize.x, (*it)*columnSize.y, offsetZ,
               x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
                           0.f,                0.f, 0.f, 
                     material);
            gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+(index++);
            gpuKernel->setPrimitive( gNbPrimitives, 
               x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ, 
               x+columnSize.x,                0.f, columnSize.z + offsetZ,
               x+columnSize.x,                0.f, offsetZ,
                           0.f,                0.f, 0.f, 
                     material);
         }

         value = (*it);
         x += columnSpacing.x;
         ++it;
      }
      material++;
   }

   gNbBoxes = gpuKernel->compactBoxes(update);

   // Post processing effects
   PostProcessingInfo postProcessingInfo = chartInfo.postProcessingInfo;
   postProcessingInfo.param2.x = 200.f;
   //postProcessingInfo.param2.x = (postProcessingInfo.type.x==0) ? sceneInfo.maxPathTracingIterations.x*10.f : 5000.f;
   //postProcessingInfo.param3.x = (postProcessingInfo.type.x != 2 ) ? 40+sceneInfo.maxPathTracingIterations.x*5 : 16;

   // Shadows
   sceneInfo.graphicsLevel.x = (postProcessingInfo.type.x == 2) ? 4 : 5;

   // Rotation
   //gpuKernel->rotatePrimitives( gRotationCenter, chartInfo.rotationAngles, 0, gNbBoxes );

   // Background color
   sceneInfo.backgroundColor = (postProcessingInfo.type.x == 2 ) ? gBkBlack : sceneInfo.backgroundColor;

   // Rendering process
   unsigned char* image = nullptr;
   for( int i(0); i<sceneInfo.maxPathTracingIterations.x; ++i)
   {
      sceneInfo.pathTracingIteration.x = i;
      gpuKernel->setPostProcessingInfo( postProcessingInfo );
      gpuKernel->setSceneInfo( sceneInfo );
      cameraAngles = chartInfo.rotationAngles;
      gpuKernel->setCamera( cameraOrigin, cameraTarget, cameraAngles );
      gpuKernel->render_begin(0.f);
      gpuKernel->render_end();
      image = gpuKernel->getBitmap();
   }
   std::string filename = "chart.jpg";
   saveToJPeg( request, filename, sceneInfo, image );
}

void buildColumnChart( Lacewing::Webserver::Request& request, ChartInfo& chartInfo, const bool& update )
{
   float3 cameraOrigin = chartInfo.viewPos;
   float3 cameraTarget = chartInfo.viewPos;
   cameraTarget.z += 5000.f;
   float3 cameraAngles = gViewAngles;

   float3 columnSize    = { 400.f, 40.f, 400.f };
   float3 columnSpacing = { 440.f, 40.f, 800.f };
   float3 size = {500.f,500.f,500.f};
   int material = 100;

   SceneInfo sceneInfo = chartInfo.sceneInfo;
   size_t len(sceneInfo.width.x*sceneInfo.height.x*gWindowDepth);
   long renderingTime = GetTickCount();

   // Ground
   float sideSize = columnSpacing.x*chartInfo.values[0].size()*0.9f;
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex;
   if(!update) gChartStartIndex = gNbPrimitives;

   gpuKernel->setPrimitive( gNbPrimitives, 
         -sideSize, -10.f, -sideSize, 
         sideSize, -10.f, -sideSize,
         sideSize, -10.f,  sideSize,
               0.f, -10.f,       0.f, 
         material);

   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+1;
   gpuKernel->setPrimitive( gNbPrimitives, 
         sideSize, -10.f,  sideSize, 
         -sideSize, -10.f,  sideSize,
         -sideSize, -10.f, -sideSize,
               0.f, 0.f,       0.f, 
         material);

   // Wall
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+2;
   gpuKernel->setPrimitive( gNbPrimitives, 
         -sideSize,         -10.f,  sideSize, 
         sideSize,         -10.f,  sideSize,
         sideSize, sideSize-10.f,  sideSize,
               0.f,      0.f,       0.f, 
         material);
      
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+3;
   gpuKernel->setPrimitive( gNbPrimitives, 
         sideSize, sideSize-10.f,  sideSize, 
         -sideSize, sideSize-10.f,  sideSize,
         -sideSize,         -10.f,  sideSize,
               0.f,      0.f,       0.f, 
         material);

   // Lamp
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptXZPlane ) : gChartStartIndex+4;
   gpuKernel->setPrimitive( gNbPrimitives,  static_cast<float>(rand()%10000-5000), 5000.f, -2000.f-static_cast<float>(rand()%5000), 2000.f, 0.f, 500.f, DEFAULT_LIGHT_MATERIAL);

   // Build Chart
   int index(0);
   for( int s(0); s<NB_MAX_SERIES; ++s )
   {
      material = 20+s*5;
      float x=-(columnSpacing.x*chartInfo.values[s].size())/2.f + columnSpacing.x/4.f;
      std::vector<float>::const_iterator it = chartInfo.values[s].begin();
      while( it != chartInfo.values[s].end() )
      {
         float offsetZ = s*columnSpacing.z - ( NB_MAX_SERIES * columnSpacing.z )/2.f;
         // Front
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+5+index;
         gpuKernel->setPrimitive( gNbPrimitives, 
                           x,                0.f, offsetZ, 
            x+columnSize.x,                0.f, offsetZ,
            x+columnSize.x, (*it)*columnSize.y, offsetZ,
                        0.f,                0.f, 0.f, 
                  material);

         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+6+index;
         gpuKernel->setPrimitive( gNbPrimitives, 
            x+columnSize.x, (*it)*columnSize.y, offsetZ,
                           x, (*it)*columnSize.y, offsetZ,
                           x,                0.f, offsetZ, 
                        0.f,                0.f, 0.f, 
               material);

         // Back
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+7+index;
         gpuKernel->setPrimitive( gNbPrimitives, 
                     x,            0.f, columnSize.z + offsetZ, 
            x+columnSize.x,            0.f, columnSize.z + offsetZ,
            x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
                     0.f,            0.f,      0.f, 
               material);

         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+8+index;
         gpuKernel->setPrimitive( gNbPrimitives, 
            x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
                     x, (*it)*columnSize.y, columnSize.z + offsetZ,
                     x,            0.f, columnSize.z + offsetZ, 
                     0.f,            0.f,      0.f, 
               material);

         // Right side
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+9+index;
         gpuKernel->setPrimitive( gNbPrimitives, 
            x+columnSize.x,                0.f,          0.f + offsetZ, 
            x+columnSize.x,                0.f, columnSize.z + offsetZ,
            x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
                     0.f,            0.f,      0.f, 
               material);

         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+10+index;
         gpuKernel->setPrimitive( gNbPrimitives, 
            x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
            x+columnSize.x, (*it)*columnSize.y,          0.f + offsetZ,
            x+columnSize.x,                0.f,          0.f + offsetZ, 
                     0.f,            0.f,         0.f, 
               material);

         // Left side
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+11+index;
         gpuKernel->setPrimitive( gNbPrimitives, 
                     x,                0.f,          0.f + offsetZ, 
                     x,                0.f, columnSize.z + offsetZ,
                     x, (*it)*columnSize.y, columnSize.z + offsetZ,
                     0.f,            0.f,      0.f, 
               material);

         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+12+index;
         gpuKernel->setPrimitive( gNbPrimitives, 
                     x, (*it)*columnSize.y, columnSize.z + offsetZ,
                     x, (*it)*columnSize.y,      0.f + offsetZ,
                     x,            0.f,      0.f + offsetZ, 
                     0.f,            0.f,      0.f, 
               material);

         // Top side
         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+13+index;
         gpuKernel->setPrimitive( gNbPrimitives, 
                     x, (*it)*columnSize.y,      0.f + offsetZ, 
            x+columnSize.x, (*it)*columnSize.y,      0.f + offsetZ,
            x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
                     0.f,            0.f,      0.f, 
               material);

         gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+14+index;
         gpuKernel->setPrimitive( gNbPrimitives, 
            x+columnSize.x, (*it)*columnSize.y, columnSize.z + offsetZ,
                     x, (*it)*columnSize.y, columnSize.z + offsetZ,
                     x, (*it)*columnSize.y,      0.f + offsetZ, 
                     0.f,            0.f,      0.f, 
               material);

         x += columnSpacing.x;
         ++it;
         index+=10;
      }
      material++;
   }

   material = 100;
   // Right Side
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+index+15;
   gpuKernel->setPrimitive( gNbPrimitives, 
         sideSize,         -10.f, -sideSize, 
         sideSize,         -10.f,  sideSize+10.f,
         sideSize, sideSize-10.f,  sideSize+10.f,
               0.f,      0.f,       0.f, 
         material);

   // Left Side
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptTriangle ) : gChartStartIndex+index+16;
   gpuKernel->setPrimitive( gNbPrimitives, 
         -sideSize,         -10.f, -sideSize, 
         -sideSize,         -10.f,  sideSize+10.f,
         -sideSize, sideSize-10.f,  sideSize+10.f,
               0.f,      0.f,       0.f, 
         material);

   gNbBoxes = gpuKernel->compactBoxes(update);

   // Post processing effects
   PostProcessingInfo postProcessingInfo = chartInfo.postProcessingInfo;
   postProcessingInfo.param2.x = 200.f;
   //postProcessingInfo.param2.x = (postProcessingInfo.type.x==0) ? sceneInfo.maxPathTracingIterations.x*10.f : 5000.f;
   //postProcessingInfo.param3.x = (postProcessingInfo.type.x != 2 ) ? 40+sceneInfo.maxPathTracingIterations.x*5 : 16;

   // Shadows
   sceneInfo.graphicsLevel.x = (postProcessingInfo.type.x == 2) ? 4 : 5;

   // Rotation
   //gpuKernel->rotatePrimitives( gRotationCenter, chartInfo.rotationAngles, 0, gNbBoxes );

   // Background color
   sceneInfo.backgroundColor = (postProcessingInfo.type.x == 2 ) ? gBkBlack : sceneInfo.backgroundColor;

   // Rendering process
   unsigned char* image = nullptr;
   for( int i(0); i<sceneInfo.maxPathTracingIterations.x; ++i)
   {
      sceneInfo.pathTracingIteration.x = i;
      gpuKernel->setPostProcessingInfo( postProcessingInfo );
      gpuKernel->setSceneInfo( sceneInfo );
      cameraAngles = chartInfo.rotationAngles;
      gpuKernel->setCamera( cameraOrigin, cameraTarget, cameraAngles );
      gpuKernel->render_begin(0.f);
      gpuKernel->render_end();
      image = gpuKernel->getBitmap();
   }
   std::string filename = "chart.jpg";
   saveToJPeg( request, filename, sceneInfo, image );
}

void renderChart( Lacewing::Webserver::Request& request, ChartInfo& chartInfo, const bool& update )
{
   switch( rand()%2 )
   {
   case 0: buildAreaChart( request, chartInfo, true ); break;
   case 1: buildColumnChart( request, chartInfo, true ); break;
   }
}

void parseChart( Lacewing::Webserver::Request& request, std::string& requestStr, const bool& update )
{
   LOG_INFO(1, "parseChart" );
   ChartInfo chartInfo;
   chartInfo.viewPos = gViewPos;
   chartInfo.rotationAngles.x = 0.f;
   chartInfo.rotationAngles.y = 0.f;
   chartInfo.rotationAngles.z = 0.f;
   chartInfo.sceneInfo = gSceneInfo;
   chartInfo.postProcessingInfo = gPostProcessingInfo;

   Lacewing::Webserver::Request::Parameter* p=request.GET();
   while( p != nullptr )
   {
      requestStr += p->Name();
      requestStr += "=";
      requestStr += p->Value();
      if( strcmp(p->Name(),"type")==0 )
      {
         // --------------------------------------------------------------------------------
         // Chart Type
         // --------------------------------------------------------------------------------
         chartInfo.chartType = atoi(p->Value());
      }
      else if ( strcmp(p->Name(),"values") == 0 )
      {
         // --------------------------------------------------------------------------------
         // values
         // --------------------------------------------------------------------------------
#if 0
         readfloats(p->Value(), chartInfo.values[0] );
#else
         for( int s(0); s<NB_MAX_SERIES; ++s) 
         {
            for( int i(0); i<10; ++i )
            {
               chartInfo.values[s].push_back(10.f+static_cast<float>(rand()%30));
            }
         }
#endif 
      }
      else if ( strcmp(p->Name(),"distance") == 0 )
      {
         // --------------------------------------------------------------------------------
         // View Distance
         // --------------------------------------------------------------------------------
         chartInfo.viewPos.z = static_cast<float>(atoi(p->Value()));
      }
      else if ( strcmp(p->Name(),"rotation") == 0 )
      {
         // --------------------------------------------------------------------------------
         // rotation angles
         // --------------------------------------------------------------------------------
         chartInfo.rotationAngles = readfloat3(p->Value());
         chartInfo.rotationAngles.x = chartInfo.rotationAngles.x/180.f*static_cast<float>(M_PI);
         chartInfo.rotationAngles.y = chartInfo.rotationAngles.y/180.f*static_cast<float>(M_PI);
         chartInfo.rotationAngles.z = chartInfo.rotationAngles.z/180.f*static_cast<float>(M_PI);
      }
      else if ( strcmp(p->Name(),"bkcolor") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Backgroud color
         // --------------------------------------------------------------------------------
         float3 c = readfloat3(p->Value());
         chartInfo.sceneInfo.backgroundColor.x = c.x/255.f;
         chartInfo.sceneInfo.backgroundColor.y = c.y/255.f;
         chartInfo.sceneInfo.backgroundColor.z = c.z/255.f;
         saturatefloat4(chartInfo.sceneInfo.backgroundColor,0.f,255.f);
      }
      else if ( strcmp(p->Name(),"quality") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Quality
         // --------------------------------------------------------------------------------
         chartInfo.sceneInfo.maxPathTracingIterations.x = atoi(p->Value());
         chartInfo.sceneInfo.maxPathTracingIterations.x = 
            (chartInfo.sceneInfo.maxPathTracingIterations.x>gMaxPathTracingIterations) ? 
            gMaxPathTracingIterations : 
            chartInfo.sceneInfo.maxPathTracingIterations.x;
      }
      else if ( strcmp(p->Name(),"size") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Image Size
         // --------------------------------------------------------------------------------
         switch( atoi(p->Value()) ) 
         {
         case  1: gWindowWidth=1024; gWindowHeight=1024; break;
         case  2: gWindowWidth=1600; gWindowHeight=1600; break;
         case  3: gWindowWidth=1920; gWindowHeight=1920; break;
         case  4: gWindowWidth=2048; gWindowHeight=2048; break;
         case  5: gWindowWidth=4096; gWindowHeight=4096; break;
         default: gWindowWidth=512;  gWindowHeight=512;  
         }
         chartInfo.sceneInfo.width.x  = gWindowWidth;
         chartInfo.sceneInfo.height.x = gWindowHeight;
      }
      else if ( strcmp(p->Name(),"postprocessing") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Post Processing
         // --------------------------------------------------------------------------------
         int postProcessing = atoi(p->Value());
         if( postProcessing<0 || postProcessing>2 ) postProcessing = 0;
         chartInfo.postProcessingInfo.type.x = postProcessing;
      }

      p = p->Next();
      if(p != nullptr) requestStr += "&";
   }

   // Render Chart
   renderChart( request, chartInfo, update );
}

void loadPDB( Lacewing::Webserver::Request& request, const MoleculeInfo& moleculeInfo )
{
   // --------------------------------------------------------------------------------
   // PDB File management
   // --------------------------------------------------------------------------------
   std::string fileName("./Pdb/");
   std::string moleculeName;
   moleculeName += ( moleculeInfo.moleculeId.length() == 0 ) ? gProteinNames[gCurrentProtein] : moleculeInfo.moleculeId;
   moleculeName += ".pdb";

   fileName += moleculeName;
   LOG_INFO(1, "Loading " << fileName );

   // Check file existence
   std::ifstream file( fileName.c_str() );
   if( file.is_open() )
   {
      file.close();
   }
   else
   {
      // If file is not in the cache, download it

      std::string url("http://www.rcsb.org/pdb/files/");
      url += moleculeName;
      HINTERNET IntOpen = ::InternetOpen("Sample", LOCAL_INTERNET_ACCESS, NULL, 0, 0);
      HINTERNET handle = ::InternetOpenUrl(IntOpen, url.c_str(), NULL, NULL, NULL, NULL);

      if( handle )
      {
         std::ofstream myfile(fileName);
         if (myfile.is_open())
         {
            request << "<p align=center>PDB File was not in the cache and had to be downloaded from <a href=http://www.rcsb.org>Protein Data Bank</a></p>";
            char buffer[2];
            DWORD dwRead=0;
            while(::InternetReadFile(handle, buffer, sizeof(buffer)-1, &dwRead) == TRUE)
            {
               if ( dwRead == 0) 
                  break;
               myfile << buffer;
            }
            myfile.close();
         }
      }
      else
      {
         // TODO!!!!
         request << "<p align=center>Unknown molecule</p>";
      }
      ::InternetCloseHandle(handle);   
   }
}

void renderPDB( Lacewing::Webserver::Request& request, const MoleculeInfo& moleculeInfo, const bool& update )
{
   float3 cameraOrigin = moleculeInfo.viewPos;
   float3 cameraTarget = moleculeInfo.viewPos;
   cameraOrigin.z += 4000.f;
   cameraTarget.z += 9000.f;
   float3 cameraAngles = gViewAngles;

   std::string fileName("./Pdb/");
   fileName += ( moleculeInfo.moleculeId.length() == 0 ) ? gProteinNames[gCurrentProtein] : moleculeInfo.moleculeId;
   fileName += ".pdb";

   // --------------------------------------------------------------------------------
   // Create 3D Scene
   // --------------------------------------------------------------------------------
   SceneInfo sceneInfo = moleculeInfo.sceneInfo;
   size_t len(sceneInfo.width.x*sceneInfo.height.x*gWindowDepth);
   long renderingTime = GetTickCount();

   // Lamp
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptSphere ) : gChartStartIndex;
   gpuKernel->setPrimitive( gNbPrimitives,  -5000.f, 5000.f, -5000.f, 50.f, 0.f, 0.f, DEFAULT_LIGHT_MATERIAL);

   if( update )
   {
      float3 objectScale = { 20.f,20.f,20.f };
      PDBReader reader;
      float3 size = reader.loadAtomsFromFile(
         fileName,*gpuKernel,
         static_cast<GeometryType>(moleculeInfo.structureType),50.f, 20.f,
         moleculeInfo.scheme,
         objectScale,false);
   }
   gNbBoxes = gpuKernel->compactBoxes(update);

   // Post processing effects
   PostProcessingInfo postProcessingInfo = moleculeInfo.postProcessingInfo;
   postProcessingInfo.param2.x = 200.f;
   //postProcessingInfo.param2.x = (postProcessingInfo.type.x==0) ? sceneInfo.maxPathTracingIterations.x*10.f : 5000.f;
   //postProcessingInfo.param3.x = (postProcessingInfo.type.x != 2 ) ? 40+sceneInfo.maxPathTracingIterations.x*5 : 16;

   // Shadows
   sceneInfo.graphicsLevel.x = (postProcessingInfo.type.x == 2) ? 4 : 5;

   sceneInfo.viewDistance.x = 100000.f;

   // Rotation
   //gpuKernel->rotatePrimitives( gRotationCenter, moleculeInfo.rotationAngles, 0, gNbBoxes );

   // Background color
   sceneInfo.backgroundColor = (postProcessingInfo.type.x == 2 ) ? gBkBlack : sceneInfo.backgroundColor;

   // Rendering process
   unsigned char* image = nullptr;
   for( int i(0); i<sceneInfo.maxPathTracingIterations.x; ++i)
   {
      sceneInfo.pathTracingIteration.x = i;
      gpuKernel->setPostProcessingInfo( postProcessingInfo );
      gpuKernel->setSceneInfo( sceneInfo );
      cameraAngles = moleculeInfo.rotationAngles;
      gpuKernel->setCamera( cameraOrigin, cameraTarget, cameraAngles );
      gpuKernel->render_begin(0.f);
      gpuKernel->render_end();
      image = gpuKernel->getBitmap();
   }
   std::string filename = "molecule.jpg";
   saveToJPeg( request, filename, sceneInfo, image );
}

void parsePDB( Lacewing::Webserver::Request& request, std::string& requestStr, const bool& update )
{
   LOG_INFO(1, "parsePDB" );
   MoleculeInfo moleculeInfo;
   moleculeInfo.moleculeId = gProteinNames[gCurrentProtein];
   moleculeInfo.structureType = 0;
   moleculeInfo.scheme=0;
   moleculeInfo.viewPos = gViewPos;
   moleculeInfo.rotationAngles.x = 0.f;
   moleculeInfo.rotationAngles.y = 0.f;
   moleculeInfo.rotationAngles.z = 0.f;
   moleculeInfo.sceneInfo = gSceneInfo;
   moleculeInfo.postProcessingInfo = gPostProcessingInfo;

   requestStr += request.GetAddress().ToString();
   requestStr += ": ";
   requestStr += request.URL();
   requestStr += "?";

   Lacewing::Webserver::Request::Parameter* p=request.GET();
   while( p != nullptr )
   {
      requestStr += p->Name();
      requestStr += "=";
      requestStr += p->Value();
      if( strcmp(p->Name(),"molecule")==0 )
      {
         // --------------------------------------------------------------------------------
         // Molecule
         // --------------------------------------------------------------------------------
         moleculeInfo.moleculeId = p->Value();
      }
      else if ( strcmp(p->Name(),"rotation") == 0 )
      {
         // --------------------------------------------------------------------------------
         // rotation angles
         // --------------------------------------------------------------------------------
         moleculeInfo.rotationAngles = readfloat3(p->Value());
         moleculeInfo.rotationAngles.x = moleculeInfo.rotationAngles.x/180.f*static_cast<float>(M_PI);
         moleculeInfo.rotationAngles.y = moleculeInfo.rotationAngles.y/180.f*static_cast<float>(M_PI);
         moleculeInfo.rotationAngles.z = moleculeInfo.rotationAngles.z/180.f*static_cast<float>(M_PI);
      }
      else if ( strcmp(p->Name(),"bkcolor") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Backgroud color
         // --------------------------------------------------------------------------------
         float3 c = readfloat3(p->Value());
         moleculeInfo.sceneInfo.backgroundColor.x = c.x/255.f;
         moleculeInfo.sceneInfo.backgroundColor.y = c.y/255.f;
         moleculeInfo.sceneInfo.backgroundColor.z = c.z/255.f;
         saturatefloat4(moleculeInfo.sceneInfo.backgroundColor,0.f,255.f);
      }
      else if ( strcmp(p->Name(),"structure") == 0 )
      {
         // --------------------------------------------------------------------------------
         // structure
         // --------------------------------------------------------------------------------
         moleculeInfo.structureType = atoi(p->Value());
         if( moleculeInfo.structureType<0 || moleculeInfo.structureType>4 ) moleculeInfo.structureType = 0;
      }
      else if ( strcmp(p->Name(),"scheme") == 0 )
      {
         // --------------------------------------------------------------------------------
         // scheme
         // --------------------------------------------------------------------------------
         moleculeInfo.scheme = atoi(p->Value());
         if( moleculeInfo.scheme<0 || moleculeInfo.scheme>2 ) moleculeInfo.scheme = 0;
      }
      else if ( strcmp(p->Name(),"quality") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Quality
         // --------------------------------------------------------------------------------
         moleculeInfo.sceneInfo.maxPathTracingIterations.x = atoi(p->Value());
         moleculeInfo.sceneInfo.maxPathTracingIterations.x = (moleculeInfo.sceneInfo.maxPathTracingIterations.x>20) ? 20 : moleculeInfo.sceneInfo.maxPathTracingIterations.x;
      }
      else if ( strcmp(p->Name(),"distance") == 0 )
      {
         // --------------------------------------------------------------------------------
         // View Distance
         // --------------------------------------------------------------------------------
         moleculeInfo.viewPos.z = static_cast<float>(atoi(p->Value()));
      }
      else if ( strcmp(p->Name(),"size") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Image Size
         // --------------------------------------------------------------------------------
         switch( atoi(p->Value()) ) 
         {
         case  1: gWindowWidth=1024; gWindowHeight=1024; break;
         case  2: gWindowWidth=1600; gWindowHeight=1600; break;
         case  3: gWindowWidth=1920; gWindowHeight=1920; break;
         case  4: gWindowWidth=2048; gWindowHeight=2048; break;
         case  5: gWindowWidth=4096; gWindowHeight=4096; break;
         default: gWindowWidth=512;  gWindowHeight=512;  
         }
         moleculeInfo.sceneInfo.width.x  = gWindowWidth;
         moleculeInfo.sceneInfo.height.x = gWindowHeight;
      }
      else if ( strcmp(p->Name(),"postprocessing") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Post Processing
         // --------------------------------------------------------------------------------
         int postProcessing = atoi(p->Value());
         if( postProcessing<0 || postProcessing>2 ) postProcessing = 0;
         moleculeInfo.postProcessingInfo.type.x = postProcessing;
      }

      p = p->Next();
      if(p != nullptr) requestStr += "&";
   }

   if( update )
   {
      // Load Molecule from file
      loadPDB( request, moleculeInfo );
   }

   // Render molecule
   renderPDB( request, moleculeInfo, update );

   // Store information about rendered molecule
   LOG_INFO(1, request.GetAddress().ToString() << " - " << request.URL() << requestStr );
   gRequests[request.GetAddress().ToString()] = requestStr;
   gNbCalls++;
}

void renderIRT( Lacewing::Webserver::Request& request, const IrtInfo& irtInfo, const bool& update )
{
   float3 cameraOrigin = irtInfo.viewPos;
   float3 cameraTarget = irtInfo.viewPos;
   cameraTarget.z += 5000.f;
   float3 cameraAngles = gViewAngles;

   std::string fileName("./irt/");
   fileName += irtInfo.filename;
   fileName += ".irt";

   // --------------------------------------------------------------------------------
   // Create 3D Scene
   // --------------------------------------------------------------------------------
   SceneInfo sceneInfo = irtInfo.sceneInfo;
   size_t len(sceneInfo.width.x*sceneInfo.height.x*gWindowDepth);
   long renderingTime = GetTickCount();

   // Lamp
   gNbPrimitives = update ? gpuKernel->addPrimitive( ptSphere ) : gChartStartIndex;
   gpuKernel->setPrimitive( gNbPrimitives,  -4000.f, 4000.f, -10000.f, 200.f, 0.f, 50.f, DEFAULT_LIGHT_MATERIAL);

   if( update )
   {
      float3 center={0.f,0.f,0.f};
      FileMarshaller fm;
      float3 size = fm.loadFromFile(*gpuKernel,fileName, center, 5000.f);
      gNbPrimitives = gpuKernel->addPrimitive( ptXZPlane );
      gpuKernel->setPrimitive( gNbPrimitives, 0.f, -2520.f, 0.f, 10000.f, 0.f, 10000.f, 100);
   }
   gNbBoxes = gpuKernel->compactBoxes(update);
      
   // Post processing effects
   PostProcessingInfo postProcessingInfo = irtInfo.postProcessingInfo;
   //postProcessingInfo.param1.x = 1000.f;//-cameraTarget.z;
   postProcessingInfo.param2.x = 200.f;
   //postProcessingInfo.param2.x = (postProcessingInfo.type.x==0) ? sceneInfo.maxPathTracingIterations.x*10.f : 5000.f;
   //postProcessingInfo.param3.x = (postProcessingInfo.type.x != 2 ) ? 40+sceneInfo.maxPathTracingIterations.x*5 : 16;

   // Shadows
   sceneInfo.graphicsLevel.x = (postProcessingInfo.type.x == 2) ? 4 : 5;

   // Rotation
   // gpuKernel->rotatePrimitives( gRotationCenter, irtInfo.rotationAngles, 0, gNbBoxes );

   // Background color
   sceneInfo.backgroundColor = (postProcessingInfo.type.x == 2 ) ? gBkBlack : sceneInfo.backgroundColor;

   // Rendering process
   unsigned char* image = nullptr;
   for( int i(0); i<sceneInfo.maxPathTracingIterations.x; ++i)
   {
      sceneInfo.pathTracingIteration.x = i;
      gpuKernel->setPostProcessingInfo( postProcessingInfo );
      gpuKernel->setSceneInfo( sceneInfo );
      cameraAngles = irtInfo.rotationAngles;
      gpuKernel->setCamera( cameraOrigin, cameraTarget, cameraAngles );
      gpuKernel->render_begin(0.f);
      gpuKernel->render_end();
      image = gpuKernel->getBitmap();
   }
   std::string filename = "irt.jpg";
   saveToJPeg( request, filename, sceneInfo, image );
}

void parseIRT( Lacewing::Webserver::Request& request, std::string& requestStr, const bool& update )
{
   LOG_INFO(1, "parseIRT" );
   IrtInfo irtInfo;
   irtInfo.viewPos = gViewPos;
   irtInfo.rotationAngles.x = 0.f;
   irtInfo.rotationAngles.y = 0.f;
   irtInfo.rotationAngles.z = 0.f;
   irtInfo.sceneInfo = gSceneInfo;
   irtInfo.postProcessingInfo = gPostProcessingInfo;

   Lacewing::Webserver::Request::Parameter* p=request.GET();
   while( p != nullptr )
   {
      requestStr += p->Name();
      requestStr += "=";
      requestStr += p->Value();
      if( strcmp(p->Name(),"model")==0 )
      {
         // --------------------------------------------------------------------------------
         // Molecule
         // --------------------------------------------------------------------------------
         irtInfo.filename = p->Value();
      }
      else if ( strcmp(p->Name(),"rotation") == 0 )
      {
         // --------------------------------------------------------------------------------
         // rotation angles
         // --------------------------------------------------------------------------------
         irtInfo.rotationAngles = readfloat3(p->Value());
         irtInfo.rotationAngles.x = irtInfo.rotationAngles.x/180.f*static_cast<float>(M_PI);
         irtInfo.rotationAngles.y = irtInfo.rotationAngles.y/180.f*static_cast<float>(M_PI);
         irtInfo.rotationAngles.z = irtInfo.rotationAngles.z/180.f*static_cast<float>(M_PI);
      }
      else if ( strcmp(p->Name(),"bkcolor") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Backgroud color
         // --------------------------------------------------------------------------------
         float3 c = readfloat3(p->Value());
         irtInfo.sceneInfo.backgroundColor.x = c.x/255.f;
         irtInfo.sceneInfo.backgroundColor.y = c.y/255.f;
         irtInfo.sceneInfo.backgroundColor.z = c.z/255.f;
         saturatefloat4(irtInfo.sceneInfo.backgroundColor,0.f,255.f);
      }
      else if ( strcmp(p->Name(),"quality") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Quality
         // --------------------------------------------------------------------------------
         irtInfo.sceneInfo.maxPathTracingIterations.x = atoi(p->Value());
         irtInfo.sceneInfo.maxPathTracingIterations.x = 
            (irtInfo.sceneInfo.maxPathTracingIterations.x>gMaxPathTracingIterations) ? 
            gMaxPathTracingIterations : 
            irtInfo.sceneInfo.maxPathTracingIterations.x;
      }
      else if ( strcmp(p->Name(),"distance") == 0 )
      {
         // --------------------------------------------------------------------------------
         // View Distance
         // --------------------------------------------------------------------------------
         irtInfo.viewPos.z = static_cast<float>(atoi(p->Value()));
      }
      else if ( strcmp(p->Name(),"size") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Image Size
         // --------------------------------------------------------------------------------
         switch( atoi(p->Value()) ) 
         {
         case  1: gWindowWidth=1024; gWindowHeight=1024; break;
         case  2: gWindowWidth=1600; gWindowHeight=1600; break;
         case  3: gWindowWidth=1920; gWindowHeight=1920; break;
         case  4: gWindowWidth=2048; gWindowHeight=2048; break;
         case  5: gWindowWidth=4096; gWindowHeight=4096; break;
         default: gWindowWidth=512;  gWindowHeight=512;  
         }
         irtInfo.sceneInfo.width.x  = gWindowWidth;
         irtInfo.sceneInfo.height.x = gWindowHeight;
      }
      else if ( strcmp(p->Name(),"postprocessing") == 0 )
      {
         // --------------------------------------------------------------------------------
         // Post Processing
         // --------------------------------------------------------------------------------
         int postProcessing = atoi(p->Value());
         if( postProcessing<0 || postProcessing>2 ) postProcessing = 0;
         irtInfo.postProcessingInfo.type.x = postProcessing;
      }

      p = p->Next();
      if(p != nullptr) requestStr += "&";
   }

   // Render
   renderIRT( request, irtInfo, update );
}

void parseURL( Lacewing::Webserver::Request& request )
{
   bool update(false);
   std::string requestStr;
   Lacewing::Webserver::Request::Parameter* p=request.GET();
   if( p )
   {
      if(!strcmp(p->Name(), "molecule"))
      {
         if( gCurrentUsecase != ucPDB  || strcmp(gCurrentUsecaseValue.c_str(),p->Value()) )
         {
            destroyKernel();
            initializeKernel(false);
            gCurrentUsecase = ucPDB;
            gCurrentUsecaseValue = p->Value();
            update=true;
         }
         parsePDB( request, requestStr, update );
      }
      else if(!strcmp(p->Name(), "model"))
      {
         if( gCurrentUsecase != ucIRT || strcmp(gCurrentUsecaseValue.c_str(),p->Value()) )
         {
            destroyKernel();
            initializeKernel(true);
            gCurrentUsecase = ucIRT;
            gCurrentUsecaseValue = p->Value();
            update=true;
         }
         parseIRT( request, requestStr, update );
      }
      else
      {
         //if( gCurrentUsecase != ucChart )
         {
            destroyKernel();
            initializeKernel(true);
            gCurrentUsecase = ucChart;
            update=true;
         }
         parseChart( request, requestStr, update );

#if 0
         FileMarshaller fm;
         fm.saveToFile( *gpuKernel, "chart.irt" );         
#endif // 0
      }
   }
   // Store information about rendered molecule
   LOG_INFO(1, request.GetAddress().ToString() << " - " << request.URL() << requestStr );
   gRequests[request.GetAddress().ToString()] = requestStr;
   gNbCalls++;
}

// 
void onGet(Lacewing::Webserver &Webserver, Lacewing::Webserver::Request &request)
{
   // --------------------------------------------------------------------------------
   // Default values
   // --------------------------------------------------------------------------------

   if (!strcmp(request.URL(), "get"))
   {
      try
      {
         parseURL( request );

#if 0
         request << "<body>";
         request << "<p align=\"center\"><b>Molecule:</b>" << moleculeId.c_str() << "</p>";
         request << "<p align=\"center\"><img border=5 bgcolor=#000000 src=\"data:image/jpg;base64,";
         request << base64_encode( (const unsigned char*)buffer, bufferLength, &len );
         request << "\"/></p>";
         request << "<p align=\"center\">Copyright(C) Cyrille Favreau</p>";
         renderingTime = GetTickCount()-renderingTime;
         request << "<p align=\"center\">Rendering time: " << renderingTime << " milliseconds on nVidia GTX 480</p>";
         /*
         request << "<p align=\"center\">molecule=XXXX 4 capital letters identifiying the molecule. The list bellow is  the only one you can use for now).<br/>";
         request << "<p align=\"center\">scheme=[0|1|2] 0: Standard, 1: Chain, 2: Residue<br/>";
         request << "<p align=\"center\">structure=[0|1|2|3] 0: Real size atoms, 1: Fixed size atoms, 2: Sticks, 3: Sticks and atoms<br/>";
         request << "<p align=\"center\">rotation=[x,y,z] Rotates the molecule according to x,y and z. Note that x,y,and z are real numbers specifying degrees of rotation for each axe.<br/>";
         request << "<p align=\"center\">quality=[1-100] Identifies the number of iterations to process. The higher the better, and slower...<br/>";
         request << "<p align=\"center\">bkcolor=[r,g,b] Specifies the red, green and blue values for background color(example: bkcolor=255,0,127)<br/>";
         request << "<p align=\"center\">postprocessing=[0|1|2] 0: None, 1: Depth of field, 2: Ambient occlusion</p>";
         request << "<p align=\"center\">Syntax: http://molecular-visualization.no-ip.org/get?molecule=XXXX[&scheme=0|1|2][&structure=0|1|2|3][&rotation=float,float,\<float\>][&quality=integer]<br/>";
         request << "<p align=\"center\">Example: http://molecular-visualization.no-ip.org/get?postprocessing=0&bkcolor=120,120,120&quality=1000&rotation=0,0,0&molecule=2M1L</p>";
         */
         request << "<p align=\"center\">Help: <a href=\"http://cudaopencl.blogspot.com\">http://cudaopencl.blogspot.com</a></p>";
         request << "<p align=\"center\"><a href=\"http://www.molecular-visualization.com\">http://www.molecular-visualization.com</a></p>";
         request << "</body>";
         delete [] buffer;
#endif // 0

         gCurrentProtein++;
         gCurrentProtein = gCurrentProtein%gProteinNames.size();
      }
      catch(...)
      {
         request << "An exception occured :-( Please try again";
      }
   }
   else
   {
      request << gNbCalls << " calls so far<br/>";
      std::map<std::string,std::string>::const_iterator iter = gRequests.begin();
      while( iter != gRequests.end() )
      {
         request << (*iter).second.c_str() << "<br/>";
         ++iter;
      }
   }
}

int main(int argc, char * argv[])
{
   Lacewing::EventPump EventPump;
   Lacewing::Webserver Webserver(EventPump);

   Webserver.onGet(onGet);
   Webserver.Host(10000);    

   initializeMolecules();

#ifdef USE_CUDA
   gpuKernel = new CudaKernel(false, 460, 0, 0);
#else
   gpuKernel = new CPUKernel(false, 460, 0, 0);
#endif
   gSceneInfo.width.x = gWindowWidth;
	gSceneInfo.height.x = gWindowHeight; 
   gSceneInfo.graphicsLevel.x = 4;
   gSceneInfo.nbRayIterations.x = 10;
   gSceneInfo.transparentColor.x = 2.f;
   gSceneInfo.viewDistance.x = 200000.f;
   gSceneInfo.shadowIntensity.x = 0.8f;
   gSceneInfo.width3DVision.x = 463.f;
   gSceneInfo.backgroundColor.x = 0.5f;
   gSceneInfo.backgroundColor.y = 0.5;
   gSceneInfo.backgroundColor.z = 0.5f;
   gSceneInfo.backgroundColor.w = 0.f;
   gSceneInfo.renderingType.x = vtStandard; //vt3DVision; // 
   gSceneInfo.renderBoxes.x = 0;
   gSceneInfo.pathTracingIteration.x = 0;
   gSceneInfo.maxPathTracingIterations.x = gTotalPathTracingIterations;
   gSceneInfo.misc = gMisc;

   gPostProcessingInfo.type.x   = ppe_none;
   gPostProcessingInfo.param1.x = 10000.f; 
   gPostProcessingInfo.param2.x = 400.f;
   gPostProcessingInfo.param3.x = 200;

   gpuKernel->setSceneInfo( gSceneInfo );
   gpuKernel->setPostProcessingInfo( gPostProcessingInfo );
   gpuKernel->initBuffers();

   EventPump.StartEventLoop();

   return 0;
}
