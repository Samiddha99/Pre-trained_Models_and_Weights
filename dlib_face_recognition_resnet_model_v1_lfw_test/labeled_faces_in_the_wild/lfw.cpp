
#include "get_lfw_pairs.h"

#include <dlib/matrix.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/svm.h>

using namespace std;
using namespace dlib;

int main()
{
    auto pairs = get_lfw_pairs();

    cout << "pairs.size(): "<< pairs.size() << endl;

    image_window win1, win2;
    for (auto& p : pairs)
    {
        if (p.are_same_person)
            cout << "SAME" << endl;
        else
            cout << "NOT SAME" << endl;

        matrix<rgb_pixel> img1, img2;
        load_image(img1, p.filename1);
        load_image(img2, p.filename2);
        win1.set_image(img1);
        win1.clear_overlay();
        win1.add_overlay(p.face1);
        win2.set_image(img2);
        win2.clear_overlay();
        win2.add_overlay(p.face2);
        cin.get();
    }
}

