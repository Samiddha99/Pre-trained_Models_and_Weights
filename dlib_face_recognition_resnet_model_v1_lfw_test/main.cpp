/*

    This program runs dlib's dlib_face_recognition_resnet_model_v1.dat model on the LFW benchmark.

*/

#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>

#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/misc_api.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include "get_lfw_pairs.h"

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------


template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;


template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

matrix<rgb_pixel> get_lfw_face_chip (
    const matrix<rgb_pixel>& img,
    const rectangle& rect,
    frontal_face_detector& detector,
    shape_predictor& sp
)
{
    rectangle best_det;
    double best_overlap = 0;
    // The face landmarking works better if the box is aligned as dlib's frontal face
    // detector would align it.  So try to report a face box that uses the detector, but if
    // one can't be found then just the default box.
    for (auto det : detector(img,-0.9))
    {
        auto overlap = box_intersection_over_union(rect, det);
        if (overlap > best_overlap)
        {
            best_det = det; 
            best_overlap = overlap;
        }
    }

    if (best_overlap < 0.3)
    {
        best_det = rect;// centered_rect(get_rect(img),100,100);
    }


    auto shape = sp(img, best_det);
    matrix<rgb_pixel> face_chip;
    extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
    return face_chip;
}


std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    std::vector<matrix<rgb_pixel>> crops; 
    // don't jitter
    //crops.push_back(img); return crops;


    thread_local random_cropper cropper;
    cropper.set_chip_dims(150,150);
    cropper.set_randomly_flip(true);
    cropper.set_max_object_height(0.99999);
    cropper.set_background_crops_fraction(0);
    cropper.set_min_object_height(0.97);
    cropper.set_translate_amount(0.02);
    cropper.set_max_rotation_degrees(3);

    std::vector<mmod_rect> raw_boxes(1), ignored_crop_boxes;
    raw_boxes[0] = shrink_rect(get_rect(img),3);

    matrix<rgb_pixel> temp; 
    for (int i = 0; i < 100; ++i)
    {
        cropper(img, raw_boxes, temp, ignored_crop_boxes);
        crops.push_back(move(temp));
    }
    return crops;
}

void test_on_lfw()
{
    const string lfw_dir = "../labeled_faces_in_the_wild";
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("../shape_predictor_68_face_landmarks.dat") >> sp;

    anet_type net;
    deserialize("../dlib_face_recognition_resnet_model_v1.dat") >> net;
    const double thresh = net.loss_details().get_distance_threshold();
    cout << "dist thresh: "<< thresh << endl;
    cout << "margin: "<< net.loss_details().get_margin() << endl;

    running_stats<double> rs, rs_pos, rs_neg;
    std::vector<double> pos_vals, neg_vals;


    std::vector<running_stats<double>> rs_folds(10);


    dlib::rand rnd;

    int cnt = 0;
    for (auto& p : get_lfw_pairs())
    {
        matrix<rgb_pixel> img, crop;
        std::vector<matrix<rgb_pixel>> crops; 
        load_image(img, lfw_dir+"/"+p.filename1);
        auto raw_image = get_lfw_face_chip(img, p.face1, detector, sp);
        std::vector<std::vector<mmod_rect>> raw_boxes(1, std::vector<mmod_rect>(1));

        crops = jitter_image(raw_image);

        matrix<float> v1 = mean(mat(net(crops)));

        load_image(img, lfw_dir+"/"+p.filename2);
        raw_image = get_lfw_face_chip(img, p.face2, detector, sp);

        crops = jitter_image(raw_image);
        matrix<float> v2 = mean(mat(net(crops)));

        double dist = length(v1-v2);
        if (p.are_same_person)
        {
            pos_vals.push_back(dist);
            if (dist < thresh)
            {
                rs.add(1);
                rs_pos.add(1);
                rs_folds[(cnt/300)%10].add(1);
            }
            else
            {
                rs.add(0);
                rs_pos.add(0);
                rs_folds[(cnt/300)%10].add(0);
            }
        }
        else
        {
            neg_vals.push_back(dist);
            if (dist > thresh)
            {
                rs.add(1);
                rs_neg.add(1);
                rs_folds[(cnt/300)%10].add(1);
            }
            else
            {
                rs.add(0);
                rs_neg.add(0);
                rs_folds[(cnt/300)%10].add(0);
            }
        }
        ++cnt;
    }
    cout << "overall lfw accuracy: "<< rs.mean() << endl;
    cout << "pos lfw accuracy: "<< rs_pos.mean() << endl;
    cout << "neg lfw accuracy: "<< rs_neg.mean() << endl;
    running_stats<double> rscv;
    for (auto& r : rs_folds) 
    {
        cout << "fold mean: " << r.mean() << endl;
        rscv.add(r.mean());
    }
    cout << "rscv.mean(): "<< rscv.mean() << endl;
    cout << "rscv.stddev(): "<< rscv.stddev() << endl;
    auto err = equal_error_rate(pos_vals, neg_vals);
    cout << "ERR accuracy: " << 1-err.first << endl;
    cout << "ERR thresh: " << err.second << endl;

    /*
    for (auto roc : compute_roc_curve(pos_vals, neg_vals))
        cout << roc.true_positive_rate << " " << roc.false_positive_rate << " " << roc.detection_threshold << endl;
    */

    cout << endl;
}

int main()
{
    test_on_lfw(); 


    /*  results with jittering

        dist thresh: 0.6
        overall lfw accuracy: 0.993833
        pos lfw accuracy: 0.994667
        neg lfw accuracy: 0.993
        fold mean: 0.995
        fold mean: 0.991667
        fold mean: 0.991667
        fold mean: 0.99
        fold mean: 0.996667
        fold mean: 0.996667
        fold mean: 0.99
        fold mean: 0.995
        fold mean: 0.996667
        fold mean: 0.995

        rscv.mean(): 0.993833
        rscv.stddev(): 0.00272732

        ERR accuracy: 0.993
        ERR thresh: 0.595778


    */

    /* results without jittering

        dist thresh: 0.6
        overall lfw accuracy: 0.988667
        pos lfw accuracy: 0.982667
        neg lfw accuracy: 0.994667
        ERR accuracy: 0.991333
        ERR thresh: 0.622864

    */
}


