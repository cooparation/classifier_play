#include "classify.h"

using caffe::Timer;

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];

  //Timer init_time;
  //init_time.Start();
  Classifier classifier(model_file, trained_file, mean_file, label_file);
  //init_time.Stop();
  //std::cout << "init time(ms): "
  //    << init_time.MilliSeconds() << std::endl;

  string file = argv[5];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  Timer total_time;
  total_time.Start();
  std::vector<Prediction> predictions = classifier.Classify(img);
  total_time.Stop();
  std::cout << "predict time(ms): "
      << total_time.MilliSeconds() << std::endl;

  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
}
