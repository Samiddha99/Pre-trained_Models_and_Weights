
#include <iostream>
#include <sstream>
#include <fstream>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <regex>

using namespace std;
using namespace dlib;


int main(int argc, char** argv)
{
    image_dataset_metadata::dataset data;

    auto truth_files = get_files_in_directory_tree("/home/davis/Downloads/training/label_2", match_ending(".txt"));

    std::regex e("label_2");
    std::regex epng("\\.txt");
    for (auto f : truth_files)
    {
        string filename = f.full_name();
        cout << f << endl;
        image_dataset_metadata::image img;
        img.filename = regex_replace(regex_replace(filename, e, "image_2"), epng, ".png");

        ifstream fin(filename);
        string label;
        matrix<double> vals;
        string line;
        while(getline(fin,line))
        {
            istringstream sin(line);
            sin >> label >> vals;
            //cout << "label: "<< label << "  " << vals << endl;
            image_dataset_metadata::box box;
            box.label = label;
            double truncated = vals(0);
            double occluded = vals(1);
            double x1 = vals(3);
            double y1 = vals(4);
            double x2 = vals(5);
            double y2 = vals(6);
            dpoint p1(x1,y1);
            dpoint p2(x2,y2);

            box.rect = rectangle(p1,p2);
            box.truncated = truncated > 0.3;
            box.occluded = occluded > 1;

            if (box.label == "DontCare" || box.truncated || box.occluded)
                box.ignore = true;

            img.boxes.push_back(box);
        }

        data.images.push_back(img);

    }

    save_image_dataset_metadata(data, "kitti_train.xml");


    return 0;

}

