/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2026-11-06
 * @author Vuzhos
 */

#include "cvlib.hpp"

namespace
{
void split_image(cv::Mat image, double stddev)
{
    // добавил, чтобы при достаточно мелком разбиении программа не вылетала
    if (image.empty())
        return;

    // считаем среднее и стандартное отклонение картинк
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    // если стандартное отклонение меньше порога, то считаем, что картинка однородна и закрашиваем все средним значением
    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        return;
    }

    // в ином случае бьем картинку на 4 сегмента
    const auto width = image.cols;
    const auto height = image.rows;

    split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), stddev);
    split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), stddev);
    split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), stddev);
    split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), stddev);
}

// функция, которая понадобиться в merge_image для поиска корневого сегмента, которому принадлежит сегмент id
int find_root(const std::vector<int>& parent, int id)
{
    while (parent[id] != id)
    {
        id = parent[id];
    }

    return id;
}

// собственно, сама функция объединения сегментов
void merge_image(cv::Mat& image, double threshold)
{
    const int rows = image.rows;
    const int cols = image.cols;

    // матрица меток (каждый сегмент имеет свою метку)
    cv::Mat labels(rows, cols, CV_32SC1, cv::Scalar(-1));

    // для каждого сегмента хранится информация об общей сумме яркости и количестве пикселей в них (по большей части для подсчета среднего значеняи яркости на сегмент)
    std::vector<double> sums;
    std::vector<int> counts;

    // при поиске пикселей, принадлежащих одному сегменту, буду смотреть 4 пикселя: верх, низ, право и лево
    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    // цикл присваивания пикселям начальных меток
    int label = 0;
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            // если пиксель уже имеет метку, то пропускаем его
            if (labels.at<int>(y, x) != -1)
                continue;

            // берем значение яркости пикселя, чтобы найти все соседние пиксели с таким же значением
            const uchar value = image.at<uchar>(y, x);

            // создаем очередь для поиска соседних пикселей (туда потом будут добавляться и соседние пиксели, и их соседи, и так далее)
            std::queue<cv::Point> queue;
            queue.push(cv::Point(x, y));

            labels.at<int>(y, x) = label;
            double sum = 0;
            int count = 0;

            // цикл поиска соседних пикселей, что лежат в одном сегменте
            while (!queue.empty())
            {
                // взяли первый пиксель с очереди
                cv::Point p = queue.front();
                queue.pop();

                // закинули его в сегмент
                sum += image.at<uchar>(p.y, p.x);
                ++count;

                // проверяем соседей
                for (int i = 0; i < 4; ++i)
                {
                    int nx = p.x + dx[i];
                    int ny = p.y + dy[i];

                    // если соседний пиксель выходит за границы картинки, то пропускаем его
                    if (nx < 0 || nx >= cols || ny < 0 || ny >= rows)
                        continue;

                    // если соседний пиксель уже имеет метку, то пропускаем его
                    if (labels.at<int>(ny, nx) != -1)
                        continue;

                    // если соседний пиксель имеет другое значение яркости, то пропускаем его
                    if (image.at<uchar>(ny, nx) != value)
                        continue;

                    // а вот если все хорошо, то присваиваем этому пикселю такую же метку и добавляем его в очередь
                    labels.at<int>(ny, nx) = label;
                    queue.push(cv::Point(nx, ny));
                }
            }

            // после того, как нашли все пиксели сегмента, сохраняем информацию о сумме яркости и количестве пикселей в нем и переходим к следующему сегменту
            sums.push_back(sum);
            counts.push_back(count);
            ++label;
        }
    }

    // создаем DSU, в котором хранятся наши сегменты
    std::vector<int> parent(label);
    for (int i = 0; i < label; ++i)
        parent[i] = i;

    // цикл объединения сегментов
    bool merged = true;
    while (merged)
    {
        merged = false;

        // итерируем все пиксели
        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                // получаем метку текущего пикселя
                const int current = labels.at<int>(y, x);

                // проверяем правого и нижнего соседа (2 достаточно)
                const int dx_check[] = {1, 0};
                const int dy_check[] = {0, 1};
                for (int i = 0; i < 2; ++i)
                {
                    int nx = x + dx_check[i];
                    int ny = y + dy_check[i];

                    // если сосед вне картинки, то пропускаем
                    if (nx >= cols || ny >= rows)
                        continue;

                    // смотрим метку соседа
                    int neighbor = labels.at<int>(ny, nx);

                    // и проверяем, лежит ли сосед в том же сегменте
                    int a = find_root(parent, current);
                    int b = find_root(parent, neighbor);

                    // если лежит, пропускаем
                    if (a == b)
                        continue;

                    // а вот если нет, то вычисляем средние яркости этих сегментов
                    double mean_a = sums[a] / counts[a];
                    double mean_b = sums[b] / counts[b];

                    // если модуль их разницы меньше предоставленного порога, то мы их объединяем
                    if (std::abs(mean_a - mean_b) <= threshold)
                    {
                        parent[b] = a;

                        sums[a] += sums[b];
                        counts[a] += counts[b];

                        merged = true;
                    }
                }
            }
        }
    }

    // наконец, каждому пикселю в изображении выдаем значение яркости, соответствующее среднему значению в сегменте, в котором он лежит
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            int root = find_root(parent, labels.at<int>(y, x));

            image.at<uchar>(y, x) = cv::saturate_cast<uchar>(sums[root] / counts[root]);
        }
    }
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev, double merge_threshold)
{
    // split part
    cv::Mat res = image;
    split_image(res, stddev);

    // merge part
    merge_image(res, merge_threshold);
    return res;
}
} // namespace cvlib
