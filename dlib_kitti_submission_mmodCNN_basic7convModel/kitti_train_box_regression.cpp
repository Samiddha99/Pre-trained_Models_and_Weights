

#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
);


template <typename T>
std::pair<double,T> find_best_matching_box(
    const std::vector<T>& boxes,
    const rectangle& rect
) 
{
    double match = 0;
    rectangle best_rect;
    for (unsigned long i = 0; i < boxes.size(); ++i)
    {
        const double new_match = box_intersection_over_union(rect, boxes[i]);
        if (new_match > match)
        {
            match = new_match;
            best_rect = boxes[i];
        }
    }

    return std::make_pair(match,best_rect);
}

// T and U are rectangle or mmod_rect
template <typename T, typename U>
std::vector<std::vector<full_object_detection>> make_bouding_box_regression_training_data (
    const std::vector<std::vector<T>>& truth,
    const std::vector<std::vector<U>>& detections
)
{
    DLIB_CASSERT(truth.size() == detections.size());
    std::vector<std::vector<full_object_detection>> result;
    for (size_t i = 0; i < truth.size(); ++i)
    {
        std::vector<full_object_detection> temp;
        for (const auto& box_ : truth[i])
        {
            // only for mmod_rect,  TODO, make code generic!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if (box_.ignore)
                continue;

            const rectangle box = box_;

            auto det = find_best_matching_box(detections[i], box);
            if (det.first > 0.5)
            {
                std::vector<point> truth_parts = {box.tl_corner(), box.tr_corner(), box.bl_corner(), box.br_corner()};
                temp.push_back(full_object_detection(det.second, truth_parts));
            }
        }
        result.push_back(std::move(temp));
    }
    return result;
}



int main(int argc, char** argv)
{
    try
    {
        if (argc != 2)
        {
            cout << "Give the path to the examples/cars directory as the argument to this" << endl;
            cout << "program.  For example, if you are in the examples folder then execute " << endl;
            cout << "this program by running: " << endl;
            cout << "   ./train_shape_predictor_ex cars" << endl;
            cout << endl;
            return 0;
        }

        const string sync_filename = "kitti_Cars_NOtesting_fullKittiTrainDataset_100x42detWin_IoU0.6";
        std::vector<std::vector<mmod_rect>> all_mmod_dets_test;
        deserialize("all_mmod_dets_test_"+sync_filename+".dat") >> all_mmod_dets_test;
        std::vector<std::vector<mmod_rect>> all_mmod_dets_train;
        deserialize("all_mmod_dets_train_"+sync_filename+".dat") >> all_mmod_dets_train;


        const std::string data_directory = argv[1];

        std::vector<array2d<unsigned char>> images_train, images_test;
        std::vector<std::vector<mmod_rect>> _cars_train, _cars_test;
        std::vector<std::vector<full_object_detection>> cars_train, cars_test;

        load_image_dataset(images_train, _cars_train, image_dataset_file(data_directory+"/kitti_train.xml").boxes_match_label("Car").boxes_match_label("DontCare").skip_empty_images());
        //load_image_dataset(images_test, _cars_test, image_dataset_file(data_directory+"/kitti_train_test.xml").boxes_match_label("Car").boxes_match_label("DontCare").skip_empty_images());

        add_image_left_right_flips(images_train, _cars_train, all_mmod_dets_train);
        //add_image_left_right_flips(images_test, _cars_test, all_mmod_dets_test);
        cars_train = make_bouding_box_regression_training_data(_cars_train, all_mmod_dets_train);
        //cars_test  = make_bouding_box_regression_training_data(_cars_test, all_mmod_dets_test);


        int cnt = 0;
        for (auto&& v : cars_train)
        {
            for (auto&& d : v)
                ++cnt;
        }
        cout << "\n\ncnt: "<< cnt << endl;


        shape_predictor_trainer trainer;

        trainer.set_feature_pool_region_padding(0.1);
        cout << "feature pool padding = " << trainer.get_feature_pool_region_padding() << endl;
        /*
        trainer.set_oversampling_amount(50);
        trainer.set_num_test_splits(50);
        */
        trainer.set_cascade_depth(80);
        cout << "cascade depth = " << trainer.get_cascade_depth() << endl;

        trainer.set_tree_depth(5);
        cout << "tree depth = " << trainer.get_tree_depth() << endl;
        trainer.set_nu(0.10);
        cout << "nu = " << trainer.get_nu() << endl;


        trainer.set_num_threads(4);
        trainer.be_verbose();

        // print baseline error
        {
            const std::vector<std::vector<double> > scales = get_interocular_distances(cars_train);
            running_stats<double> rs;
            for (unsigned long i = 0; i < cars_train.size(); ++i)
            {
                for (unsigned long j = 0; j < cars_train[i].size(); ++j)
                {
                    const double scale = scales[i][j]; 

                    rectangle box = cars_train[i][j].get_rect();
                    std::vector<point> det_parts = {box.tl_corner(), box.tr_corner(), box.bl_corner(), box.br_corner()};
                    for (unsigned long k = 0; k < cars_train[i][j].num_parts(); ++k)
                    {
                        if (cars_train[i][j].part(k) != OBJECT_PART_NOT_PRESENT)
                        {
                            double score = length(det_parts[k] - cars_train[i][j].part(k))/scale;
                            rs.add(score);
                        }
                    }
                }
            }
            cout << "untrained accuracy on training data: "<< rs.mean() << endl;
        }

        // Now finally generate the shape model
        shape_predictor sp = trainer.train(images_train, cars_train);



        cout << "mean training error: "<< 
            test_shape_predictor(sp, images_train, cars_train, get_interocular_distances(cars_train)) << endl;

        //cout << "mean testing error:  "<< 
        //    test_shape_predictor(sp, images_test, cars_test, get_interocular_distances(cars_test)) << endl;

        // Finally, we save the model to disk so we can use it later.
        string filename = "sp_kitti_" + to_string(time(0)) + ".dat";
        cout << filename << endl;
        serialize(filename) << sp;
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

double interocular_distance (
    const full_object_detection& det
)
{
    rectangle r;
    for (unsigned long i = 0; i < det.num_parts(); ++i)
        r += det.part(i);

    return sqrt(r.area());
}

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
)
{
    std::vector<std::vector<double> > temp(objects.size());
    for (unsigned long i = 0; i < objects.size(); ++i)
    {
        for (unsigned long j = 0; j < objects[i].size(); ++j)
        {
            temp[i].push_back(interocular_distance(objects[i][j]));
        }
    }
    return temp;
}

// ----------------------------------------------------------------------------------------

