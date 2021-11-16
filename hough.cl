__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE| //Natural coordinates
                                CLK_ADDRESS_CLAMP_TO_EDGE| //Clamp to zeros
                                CLK_FILTER_NEAREST;



__kernel void sobel_image2D(__read_only image2d_t image,__write_only image2d_t imageOut,  int w, int h, int t1, int t2){
    int iX= get_global_id(0);
    int iY= get_global_id(1);
    int dif, av_dif;

    if((iX>= 0)&&(iX< w) && (iY>= 0)&&(iY< h)) {
        uint4 pixelV= read_imageui( image, sampler, (int2)(iX,iY)); //uint4 stores 4 values of unsigned integers

        //top left
        uint4 a = read_imageui(image, sampler, (int2)(iX,iY)); //get RGB from each position
        //top center
        uint4 b = read_imageui(image, sampler, (int2)(iX+1,iY));
        //top right
        uint4 c = read_imageui(image, sampler, (int2)(iX+2,iY));
        //center left
        uint4 d  = read_imageui(image, sampler, (int2)(iX,iY+1));
        //center right
        uint4 f  = read_imageui(image, sampler, (int2)(iX+2,iY+1));
        //bottom left
        uint4 g = read_imageui(image, sampler, (int2)(iX,iY+2));
        //bottom center
        uint4 h  = read_imageui(image, sampler, (int2)(iX+1,iY+2));
        //botom right
        uint4 i =read_imageui(image, sampler, (int2)(iX+2,iY+2));
        //calc Sx
        int4 Sx =convert_int4((a+2*d+g) - (c+2*f+i)); //int4 stores separate values of R, G and B
        //calc Sy
        int4 Sy =convert_int4((g+2*h+i) - (a+2*b+c));
        //get S
        uint4 S = (abs(Sx) + abs(Sy));

        //get diff between componentes
        dif = (abs(S.x-S.y)+abs(S.y-S.z)+abs(S.x-S.z));
        av_dif = (abs(S.x-S.y)+abs(S.y-S.z)+abs(S.x-S.z)/3);

        // if more than t1 and t2 white else black
        if(t1< dif && t2< av_dif /*&& (iY>= 120)*/){
            write_imageui( imageOut, (int2)(iX,iY) , (uint4)(255, 255, 255, 0));
        }
        else{
            write_imageui( imageOut, (int2)(iX,iY) , (uint4)( 0, 0, 0,0));
        }



    }
}

__kernel void line_seg(__global uchar* image, int w, int h, int padding, int threshold, __global int* votes_positive,  __global int* votes_positive_1, __global uchar* imageOut){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * (w*3 ) + x*3 ;
    double theta, theta_1 ;
    double rho, rho_1;
    int place, place_1;

    if(((x < w) && (y < h)) && ( image[idx] == threshold)) {// check if x and y are valid image coordinates
        if(y > 30) {
            for(int i = 45; i < 89; i++){

                theta = i * M_PI/180;
                rho = x * cos(theta) + y * sin(theta);
                place = (int)(rho * 180 + theta);

                atomic_add( &votes_positive[place], 1);
            }
            for(int i = 91; i < 130; i++){

                theta_1 = i * M_PI/180;
                rho_1 = x * cos(theta_1) + y * sin(theta_1);
                place_1 = (int)(rho_1 * 180 + theta_1);

                atomic_add( &votes_positive_1[place_1], 1);
            }
        }
    }
}

__kernel void line_seg_image2D( __read_only image2d_t image, int w, int h, int padding, int threshold, __global int* votes_positive,  __global int* votes_positive_1){
     int iX= get_global_id(0);
    int iY= get_global_id(1);
    int dif, av_dif;
    double theta, theta_1 ;
    int rho, rho_1;
    int place, place_1;

    if((iX>= 0)&&(iX< w) && (iY>= 0)&&(iY< h)) {
        uint4 pixelV= read_imageui( image, sampler, (int2)(iX,iY));

        if((iY>= 100) && (iY <= 300) && (iX < 450) &&(pixelV.y >= threshold)) {
            for(int i = 45; i < 65; i++){
                //iY>= 150) && (iY <= 400)&& (iX < 550) &&
                //(iY > 150) && (iX < 550) && pixelV.y >= threshold)
                theta = i * M_PI/180;
                rho = (int)(iX * cos(theta) + iY * sin(theta) + 948/2);

                atomic_add( &votes_positive[(rho*180 + i)], 1);
            }
        }
        if(( (iY>= 100)&& (iY <= 320)&& (iX >= 375) && pixelV.y >= threshold)) {
           for(int a = 115; a < 170; a++){
                //(iY > 175) && (iX > 550)
                //(iY>= 200)&& (iY <= 700)&& (iX >= 375) &&
                theta_1 = a * M_PI/180;
                rho_1 = (int)(iX * cos(theta_1) + iY * sin(theta_1) + 948/2);

                atomic_add( &votes_positive_1[(rho_1*180 + a)], 1);
            }
        }
    }
}


__kernel void grey_image2D(__read_only image2d_t image,__write_only image2d_t imageOut,  int w, int h){
    int iX= get_global_id(0);
    int iY= get_global_id(1);


    if((iX>= 0)&&(iX< w) && (iY>= 0)&&(iY< h)) {
        uint4 pixelV= read_imageui( image, sampler, (int2)(iX,iY));
        int a = (pixelV.x+pixelV.y+pixelV.z)/3;
        write_imageui( imageOut, (int2)(iX,iY) , (uint4)(a,a,a,0));

    }
}

__kernel void binary_threshold_image2D(__read_only image2d_t image,__write_only image2d_t imageOut,  int w, int h, int threshold){
    int iX= get_global_id(0);
    int iY= get_global_id(1);
    uint4 neg = (uint4)(255, 255, 255 , 0); //alpha must remain the same(255)

    if((iX>= 0)&&(iX< w) && (iY>= 0)&&(iY< h)) {
     uint4 pixelV= read_imageui( image, sampler, (int2)(iX,iY));
        if(pixelV.x >= threshold){
            write_imageui( imageOut, (int2)(iX,iY) , (uint4)(255, 255, 255, 0));
        }
        else{
            write_imageui( imageOut, (int2)(iX,iY) , (uint4)( 0, 0, 0,0));
        }

    }
}

__kernel void selection(__global uchar* image, int w, int h, int padding, int threshold, __global int* selection, __global uchar* imageOut){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * (w*3 + padding) + x*3 ;


    if(((x < w) && (y < h)) && ( image[idx] > threshold)) {// check if x and y are valid image coordinates

    }
}



__kernel void after_selection(__global uchar* image, int w, int h, int padding, __global int* selection, __global uchar* imageOut){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * (w*3 + padding) + x*3 ;
    double theta, theta_1 ;
    double rho, rho_1;
    int place, place_1;


        if(y > 30) {
            for(int i = 35; i < 89; i++){

                theta = i * M_PI/180;
                rho = x * cos(theta) + y * sin(theta);
                place = rho * 180 + theta;


            }
            for(int a = 91; a < 140; a++){

                theta_1 = a * M_PI/180;
                rho_1 = x * cos(theta_1) + y * sin(theta_1);
                place_1 = rho_1 * 180 + theta_1;

            }
        }

}