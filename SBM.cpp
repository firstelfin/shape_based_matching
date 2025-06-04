#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
using namespace std;
using namespace cv;

static std::string prefix = "/data1/2024_project/stategrid/template_match/shape_based_matching/test";

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

}

void shape_base_match(Mat search_img, Mat templ, int& ssim, int num_features = 128, vector<int>(pyramid_levels) = {4, 8}, bool use_rot = true){
    ssim = 0;
    Timer timer;
    line2Dup::Detector detector(num_features, pyramid_levels);  // num_features=128, pyramid_levels = 2, T_at_level = (4, 8)
    // timer.out("init detector");
    // 加载Templ模板的特征
    Mat mask = Mat(templ.size(), CV_8UC1, {255});
    // padding to avoid rotating out
    int padding1 = 100;
    cv::Mat padded_templ = cv::Mat(templ.rows + 2*padding1, templ.cols + 2*padding1, templ.type(), cv::Scalar::all(0));
    templ.copyTo(padded_templ(Rect(padding1, padding1, templ.cols, templ.rows)));

    cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding1, mask.cols + 2*padding1, mask.type(), cv::Scalar::all(0));
    mask.copyTo(padded_mask(Rect(padding1, padding1, templ.cols, templ.rows)));

    shape_based_matching::shapeInfo_producer shapes(padded_templ, padded_mask);
    shapes.angle_range = {-40, 40};
    shapes.angle_step = 1;

    shapes.scale_range = {1}; // support just one
    shapes.produce_infos();

    std::vector<shape_based_matching::Info> infos_have_templ;
    string class_id = "test";

    bool is_first = true;

    // for other scales you want to re-extract points: 
    // set shapes.scale_range then produce_infos; set is_first = false;

    int first_id = 0;
    float first_angle = 0;
    for(auto& info: shapes.infos){
        int templ_id;
        if(is_first){
            templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
            first_id = templ_id;
            first_angle = info.angle;

            if(use_rot) is_first = false;
        }else{
            templ_id = detector.addTemplate_rotate(
                class_id, first_id, info.angle-first_angle, {shapes.src.cols/2.0f, shapes.src.rows/2.0f}
            );
        }
        if(templ_id != -1){
            infos_have_templ.push_back(info);
        }
    }
    // timer.out("add templ end");
    
    // 开始搜索Templ的匹配分数
    std::vector<std::string> ids;
    ids.push_back("test");
    assert(!search_img.empty() && "search_img is empty!");
    int padding = 250;
    cv::Mat padded_img = cv::Mat(
        search_img.rows + 2*padding,
        search_img.cols + 2*padding, 
        search_img.type(), cv::Scalar::all(0)
    );
    search_img.copyTo(padded_img(Rect(padding, padding, search_img.cols, search_img.rows)));

    int stride = 16;
    int n = padded_img.rows/stride;
    int m = padded_img.cols/stride;
    Rect roi(0, 0, stride*m , stride*n);
    Mat img = padded_img(roi).clone();
    assert(img.isContinuous());

    
    auto matches = detector.match(img, 90, ids);
    // timer.out("match end");

    if(img.channels() == 1) cvtColor(img, img, cv::COLOR_GRAY2BGR);

    size_t top5 = 1;
    if(top5>matches.size()) top5=matches.size();
    for(size_t i=0; i<top5; i++){
        auto match = matches[i];
        auto templ = detector.getTemplates(
            "test",
            match.template_id);

        // 270 is width of template image
        // 100 is padding when training
        // tl_x/y: template croping topleft corner when training

        float r_scaled = 270/2.0f*infos_have_templ[match.template_id].scale;

        // scaling won't affect this, because it has been determined by warpAffine
        // cv::warpAffine(src, dst, rot_mat, src.size()); last param
        float train_img_half_width = 270/2.0f + 100;
        float train_img_half_height = 270/2.0f + 100;

        // center x,y of train_img in test img
        float x =  match.x - templ[0].tl_x + train_img_half_width;
        float y =  match.y - templ[0].tl_y + train_img_half_height;

        cv::Vec3b randColor;
        randColor[0] = rand()%155 + 100;
        randColor[1] = rand()%155 + 100;
        randColor[2] = rand()%155 + 100;
        for(int i=0; i<templ[0].features.size(); i++){
            auto feat = templ[0].features[i];
            cv::circle(img, {feat.x+match.x, feat.y+match.y}, 3, randColor, -1);
        }

        cv::putText(img, to_string(int(round(match.similarity))),
                    Point(match.x+r_scaled-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);

        cv::RotatedRect rotatedRectangle({x, y}, {2*r_scaled, 2*r_scaled}, -infos_have_templ[match.template_id].angle);

        cv::Point2f vertices[4];
        rotatedRectangle.points(vertices);
        for(int i=0; i<4; i++){
            int next = (i+1==4) ? 0 : (i+1);
            cv::line(img, vertices[i], vertices[next], randColor, 2);
        }
        std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
        std::cout << "match.similarity: " << match.similarity << std::endl;
        ssim = match.similarity;
        // return int(round(match.similarity));
    }
    timer.out("match end");

    imshow("img", img);
    if (cv::waitKey(10000) == 10000) cv::destroyAllWindows();

    std::cout << "test end" << std::endl << std::endl;
    
    // return 0;
}

int main(){
    // angle_test("test", true); // test or train
    // template_linear_memories();
    cv::Mat search_img = imread(prefix+"/case1/test.png");
    cv::Mat Templ_img = imread(prefix+"/case1/train.png");
    Rect roi(130, 110, 270, 270);
    cv::Mat templ_img = Templ_img(roi).clone();
    int ssim;
    int &ssim2 = ssim;
    shape_base_match(search_img, templ_img, ssim2);
    cout << "ssim:" << ssim << endl;
    return 0;
}
