#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int width, height;
Mat hsv, thresh, image;

struct pt {
	int x, y;
	pt(int _x, int _y) {
		x = _x;
		y = _y;
	}

	pt() {
		x = 0;
		y = 0;
	}
};

//int img[height][width][3];
//bool threshed[height][width];
uchar *threshed;

bool vis[1000][1000];

uchar val(pt p) {
	return thresh.data[p.y * width + p.x];
}

bool edge(pt p1, pt p2) {
	return val(p1) == val(p2) && !vis[p2.x][p2.y];
}

pt q[1000 * 1000];
pt *head = &q[0], *tail = &q[0];

void push(pt c) {
	*tail = c;
	tail++;
}

void pop() {
	head++;
}

pt front() {
	return *head;
}

bool empty() {
	return head == tail;
}

void reset() {
	head = tail = &q[0];
}

struct comp {
	int npixels;
	pt min, max;
	pt sum;
	comp() {
		npixels = 0;
		min = pt(1 << 30, 1 << 30);
		max = pt(-1, -1);
		sum = pt(0, 0);
	}

	comp(pt p) {
		npixels = 1;
		min = p;
		max = p;
		sum = p;
	}

};

pt min_pt(pt p, pt q) {
	return pt(min(p.x, q.x), min(p.y, q.y));
}

pt max_pt(pt p, pt q) {
	return pt(max(p.x, q.x), max(p.y, q.y));
}

void mergec(comp *c, comp d) {
	c->npixels += d.npixels;
	c->min = min_pt(c->min, d.min);
	c->max = max_pt(c->max, d.max);
	c->sum = pt(c->sum.x + d.sum.x, c->sum.y + d.sum.y);
}

comp bfs(pt start) {
	reset();
	push(start);
	
	comp ret;

	int n = 0;
	while (!empty()) {
		pt cur = front();
		int x = cur.x, y = cur.y;
		pop();
		
		if (vis[x][y]) continue;
		vis[x][y] = true;
		mergec(&ret, comp(cur));
		//printf("%d\n", threshed);
		if (x >= 1 && edge(cur, pt(x - 1, y))) push(pt(x - 1, y));
		if (y >= 1 && edge(cur, pt(x, y - 1))) push(pt(x, y - 1));
		if (x < width - 1 && edge(cur, pt(x + 1, y))) push(pt(x + 1, y));
		if (y < height - 1 && edge(cur, pt(x, y + 1))) push(pt(x, y + 1));
	}
	return ret;
}

int max(int x, int y) {
	return (x < y) ? y : x;
}

double compdist_sq(comp c, comp d) {
	pt c_cent = pt((double)c.sum.x / (double)c.npixels, (double)c.sum.y / (double)c.npixels);

	pt d_cent = pt((double)d.sum.x / (double)d.npixels, (double)d.sum.y / (double)d.npixels);
	pt diff = pt(c_cent.x - d_cent.x, c_cent.y - d_cent.y);
	return diff.x * diff.x + diff.y * diff.y;
}

int num_components() {
	memset(vis, 0, sizeof(vis));
	vector<comp> hori, vert;
	//int ret = 0;
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			if (!vis[x][y] && val(pt(x, y))) {
				//ret++;
				//printf("%d %d\n", x, y);
				//if (bfs(pt(x, y)) >= 0) ret++;
				comp c = bfs(pt(x, y));
				if (c.npixels > 200) {
					double aspect = (double)(c.max.y - c.min.y) / (double)(c.max.x - c.min.x);
					if (aspect > 1) {
						/*rectangle(image, Point(c.min.x, c.min.y), Point(c.max.x, c.max.y), Scalar(255, 0, 0));
						printf("horizontal: %d\n", c.npixels);*/
						hori.push_back(c);
					} else if (aspect < 1) {
						/*rectangle(image, Point(c.min.x, c.min.y), Point(c.max.x, c.max.y), Scalar(0, 255, 0));
						printf("vertical: %d\n", c.npixels);*/
						vert.push_back(c);
					}
				}
			}
		}
	}
	vector<int> mate(hori.size());
	for (int i = 0; i < hori.size(); i++) {
		int m = 0;
		for (int j = 0; j < vert.size(); j++) {
			if (compdist_sq(hori[i], vert[j]) < compdist_sq(hori[i], vert[m])) {
				j = m;
			}
		}
		//comp c2 = hori[i];
		//mergec(&c2, vert[m]);

		//rectangle(image, Point(c2.min.x, c2.min.y), Point(c2.max.x, c2.max.y), Scalar(0, 0, 255));
	}

	//return ret;
	return 0;
}

unsigned char _hsl_lo[3] = {0, 100, 75};
unsigned char _hsl_hi[3] = {100, 255, 255};
vector<unsigned char> hsl_lo(_hsl_lo, _hsl_lo + 3);
vector<unsigned char> hsl_hi(_hsl_hi, _hsl_hi + 3);

int main(int argc, char** argv) {
	image = imread("/Users/sujit/Documents/Metro4/FIRST/HotTargetWithLEDs/image.jpg");
	cvtColor(image, hsv, COLOR_BGR2HSV);
	inRange(hsv, hsl_lo, hsl_hi, thresh);
	//namedWindow("test", WINDOW_AUTOSIZE);
	width = thresh.cols;
	height = thresh.rows;
	//num_components();
	//imshow("test", image);
	//while (true) waitKey(0);
	//printf("%d\n", num_components());
	for (int i = 1; i < 150; i++) {
		//num_components();
		printf("%d\n", num_components());
	}
	return 0;
}
