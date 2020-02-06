#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
#include <assert.h>
#include "utils.h"
#include "HGNNet.h"
#include "HGBoard.h"
//#include "OTGBoard.h"

using namespace std;

typedef const char cchar;

#define		PI			3.1415926536

struct DataItem {
public:
	DataItem(bool bturn = true, const string& ktext = string(), double score = 0)
		: m_bturn(bturn)
		, m_ktext(ktext)
		, m_score(score)
	{
	}
public:
	bool		m_bturn;		//	black turn
	string		m_ktext;
	double	m_score;
};

void test_genMoves();
void test_randomPlayOut();
void test_expctScoreRPO();
void test_expctScoreRPO2();
void test_readData();
void test_linearFuncArg1();			//	y = 2*x - 1、x: [-1, +1]
void test_linearFunc();			//	y = 3*x1 - 2*x2 + 1、x1,x2: [-1, +1]
//void test_linearFuncAF();			//	y = 3*x1 - 2*x2 + 1、x1,x2: [-1, +1]、すべての活性化関数
void test_sinFunc();				//	sin(2πx) を学習、x: [-1, +1]
void test_HGNNet();
void test_NNdiff();
void test_121();
void test_1201();
void test_1221();
void test_120201();
void test_12221();
void test_2argsFunc();	//	f(x, y) = { R = sqrt(x*x + y*y) + 1e-4; return sin(R) / R
void test_ReLU();			//	ランダムプレイアウトによるスコア期待値を学習 での発散テスト
void test_ReLU2();			//	ランダムプレイアウトによるスコア期待値を学習 での発散テスト
void genDataRPO();		//	ランダムプレイアウトにより 状態 → 期待スコア学習データ生成
void genDataNN();			//	学習済みNNプレイアウトにより 状態 → 期待スコア学習データ生成
void test_learnNNPO(bool verbose = true);		//	ランダムプレイアウトによるスコア期待値を学習
void test_learnRPO(bool verbose = true);		//	ランダムプレイアウトによるスコア期待値を学習
void test_learnRPO10();		//	ランダムプレイアウトによるスコア期待値を学習
void otg_genDataRPO();		//	ランダムプレイアウトにより 状態 → 期待スコア学習データ生成
void test_load_save();
void test_negaMax1();
void test_negaMax1_prime();			//	とある局面でのチェック
void test_negaMax1_random();		//	negaMax1 対 ランダム
void test_negaMax1_random_stat(int N_GAME = 100);		//	negaMax1 対 ランダム 統計

void test_OTGBoard();

int main()
{
	//test_genMoves();
	//test_randomPlayOut();
	//test_expctScoreRPO();
	//test_expctScoreRPO2();
	//test_readData();
	//test_HGNNet();
	//test_linearFuncArg1();
	//test_linearFunc();
	//##test_linearFuncAF();
	//test_sinFunc();
	//test_NNdiff();
	//test_121();
	//test_1201();
	//test_1221();
	//test_120201();
	//test_12221();
	//test_2argsFunc();
	test_learnNNPO(false);
	//test_learnRPO(false);
	//test_learnRPO10();
	//test_ReLU();
	//test_ReLU2();
	//test_load_save();
	//
	//genDataRPO();
	//genDataNN();
	//otg_genDataRPO();
	//
	//test_negaMax1();
	//test_negaMax1_prime();
	//test_negaMax1_random();
	//test_negaMax1_random_stat(1000);
	//
	//test_OTGBoard();
	//HGBoard bd;
	//cout << bd.text() << endl;
#if	0
	ofstream of("test.txt");
	of << "test\n";
	of.close();
#endif
}
void test_NNdiff()
{
	HGNNet nn;
#if	0
	cout << "# node of layers: {2 1}, tanh:\n\n";
	nn.init(vector<int>{2}, TANH);
	//nn.m_layers[0][0].m_weight[0] = 1.0;
	//nn.m_layers[0][0].m_weight[1] = -1.0;
	//nn.m_layers[0][0].m_weight[2] = 0.5;
	vector<double> input = {0.5, -0.5};
#elif	1
	cout << "# node of layers: {1 2 2 1}, ReLU:\n\n";
	nn.init(vector<int>{1, 2, 2}, RELU);
	vector<double> input = {0.5};
#else
	cout << "# node of layers: {1, 2 1}, ReLU:\n\n";
	nn.init(vector<int>{1, 2}, RELU);
	vector<double> input = {0.5};
#endif
	cout << nn.dump() << "\n";
	double T = 0.25;		//	教師値
	double y = nn.predict(input);
	double err = y - T;
	double L = err * err / 2;
	cout << "y = " << y << ", err = " << err << ", L = " << L << "\n";
	//cout << nn.predict(input) << "\n";
	nn.calcError(input, T);
	double dLdW = nn.m_layers[0][0].m_err * input[0];
	cout << "dLdW[0][0][0] = " << dLdW << endl;
	//double delta = 0.01;
	nn.m_layers[0][0].m_weight[0] += 0.01;
	cout << L << " + 0.01*" << dLdW << " = " << L + dLdW*0.01 << "\n";
	cout << nn.dump() << "\n";
	{
		double y = nn.predict(input);
		double err = y - T;
		double L = err * err / 2;
		cout << "y = " << y << ", err = " << err << ", L = " << L << "\n";
	}
}
#if	0
void otg_doRandomTurn(OTGBoard& bd, bool bt, int tcnt, int& d1, int& d2, Moves& mvs)
{
	//int d1, d2;
	do {
		d1 = g_mt() % 3;
		d2 = g_mt() % 3;
	} while (d1 == d2 && tcnt == 1);
	MovesList lst;
	//Moves mvs;
	if (bt) {
		bd.b_genMovesList(lst, d1, d2);
		if (!lst.empty()) {
			mvs = lst[g_mt() % lst.size()];
			bd.b_move(mvs);
		}
	}
	else {
		bd.w_genMovesList(lst, d1, d2);
		if (!lst.empty()) {
			mvs = lst[g_mt() % lst.size()];
			bd.w_move(mvs);
		}
	}
}
void test_OTGBoard()
{
	OTGBoard bd;
	//cout << bd.text() << "\n";
	//
#if	0
	MovesList lst;
	bd.b_genMovesList(lst, 0, 2);
	for(auto& mvs: lst) {
		for(auto& mv: mvs) cout << mv.text() << " ";
		cout << endl;
	}
#endif
	const int N_RPO = 100;
	bool bt = true;
	int d1, d2;
	Moves mvs;
	for (int cnt = 1;; ++cnt, bt = !bt) {
		cout << bd.text() << "\n";
		cout << (bt?"b ":"w ") << bd.ktext() << " ";
		if( bt )
			cout << bd.b_expctScoreRPO(N_RPO) << endl << endl;
		else
			cout << bd.w_expctScoreRPO(N_RPO) << endl << endl;
		otg_doRandomTurn(bd, bt, cnt, d1, d2, mvs);
		cout << cnt << ") " << d1 << d2 << " ";
		for(auto& mv: mvs) cout << mv.text() << " ";
		cout << endl;
		if( bd.result() != 0 ) break;
	}
	cout << bd.text() << "\n";
}
#endif
bool readData(vector<DataItem>& data, cchar* fname)
{
	//data.clear();
	string bw, ktext;
	double score;
	ifstream ifs(fname);
	while( !ifs.eof() ) {
		ifs >> bw;
		if( bw == "b" || bw == "w" ) {
			ifs >> ktext >> score;
			data.push_back(DataItem(bw == "b", ktext, score));
		}
	}
	return true;
}
void doRandomTurn(HGBoard& bd, bool bt, int tcnt)
{
	int d1, d2;
	do {
		d1 = g_mt() % 3 + 1;
		d2 = g_mt() % 3 + 1;
	} while (d1 == d2 && tcnt == 1);
	MovesList lst;
	Moves mvs;
	if (bt) {
		bd.b_genMovesList(lst, d1, d2);
		if (!lst.empty()) {
			mvs = lst[g_mt() % lst.size()];
			bd.b_move(mvs);
		}
	}
	else {
		bd.w_genMovesList(lst, d1, d2);
		if (!lst.empty()) {
			mvs = lst[g_mt() % lst.size()];
			bd.w_move(mvs);
		}
	}
}
double calcRMS_RPO(HGNNet& nn, int N_GAME = 10)
{
	HGBoard bd;
	const int N_RPO = 100;
	int nt = 0;		//	テスト数
	double sum2 = 0;
	vector<double> input(HG_NN_INSIZE);
	for (int i = 0; i < N_GAME; ++i) {
		cout << ".";
		bd.init();
		bool bt = true;
		for (int tcnt = 1;; ++tcnt, bt = !bt) {
			if( tcnt != 1 ) {
				if( bt ) {
					auto sc = bd.b_expctScoreRPO(N_RPO);
					bd.setInputNmlz(input);
					auto ps = nn.predict(input);
					sum2 += (sc - ps) * (sc - ps);
				} else {
					HGBoard b2(bd.m_white, bd.m_black);		//	白黒反転盤面
					auto sc = b2.b_expctScoreRPO(N_RPO);		//	[-3, +3]
					b2.setInputNmlz(input);
					auto ps = nn.predict(input);
					sum2 += (sc - ps) * (sc - ps);
				}
				++nt;
			}
			doRandomTurn(bd, bt, tcnt);
			if( bd.result() != 0 ) break;
		}
	}
	cout << "\n";
	return sqrt(sum2/nt);
}
void test_ReLU2()
{
	HGBoard bd;
	bool bt = true;
	string ktext = "0430000000010043010000000000";
	double T = -1.046;
	vector<double> input(HG_NN_INSIZE);
	bd.set(ktext);
	cout << bd.text();
	cout << (bt?"b ":"w ") << ktext << " " << T << "\n\n";
	bd.setInput(input);
	cout << "input: ";
	for(auto x: input) cout << x << " ";
	cout << endl;
	//
	HGNNet nn;
#if	0
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " 1}, ReLU\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE}, RELU);
#else
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << 20 << " " << 20 << " 1}, ReLU\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, 20, 20}, RELU);
#endif
	//
	double y0 = nn.predict(input);
	double L0 = (y0 - T) * (y0 - T) / 2;
	cout << "y0 = " << y0 << ", L0 = " << L0 << "\n";
//#if	1
	nn.calcError(input, T);
	cout << nn.dumpBP() << "\n";
	double dLdW = nn.m_layers[0][0].m_err * input[1];
	cout << "dLdW[0][0][1] = " << dLdW << endl;
	nn.m_layers[0][0].m_weight[1] += 0.01;
	cout << L0 << " + 0.01*" << dLdW << " = " << L0 + dLdW*0.01 << "\n";
	//cout << nn.dump() << "\n";
	{
		double y = nn.predict(input);
		double L = (y - T) * (y - T) / 2;
		cout << "y = " << y << ", L = " << L << "\n";
	}
//#else
	for (int i = 0; i < 10; ++i) {
		nn.train(input, T, 0.001);
		double y = nn.predict(input);
		double L = (y - T) * (y - T) / 2;
		cout << "y = " << y << ", L = " << L << "\n";
	}
	//cout << nn.dumpBP() << "\n";
//#endif
}
void test_ReLU()
{
	const double ALPHA = 0.001;
	HGBoard bd;
	//
	HGNNet nn;
#if	0
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << 10 << " 1}, ReLU\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, 10}, RELU);
#else
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " 1}, ReLU\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE}, RELU);
#endif
	vector<double> input(HG_NN_INSIZE);
	vector<DataItem> data;
	readData(data, "data/RPO100-001.txt");
	readData(data, "data/RPO100-002.txt");
	readData(data, "data/RPO100-003.txt");
	readData(data, "data/RPO100-004.txt");
	readData(data, "data/RPO100-005.txt");
	readData(data, "data/RPO100-006.txt");
	readData(data, "data/RPO100-007.txt");
	readData(data, "data/RPO100-008.txt");
	readData(data, "data/RPO100-009.txt");
	readData(data, "data/RPO100-010.txt");
	if( data.empty() ) {
		cout << "can't open data file.\n";
		return;
	}
	for (int i = 0; i < 40; ++i) {
		std::shuffle(data.begin(), data.end(), g_mt);
		int cnt = 0;
		for(const auto& di: data) {
			++cnt;
			//if (cnt == 4) {
			//	cout << nn.dump() << endl;
			//	cout << nn.dumpBP() << endl;
			//}
			if( di.m_bturn ) {
				bd.set(di.m_ktext);
				bd.setInput(input);
				//bd.setInputNmlz(input);
				nn.learn(input, di.m_score, ALPHA);
			} else {
				bd.set(di.m_ktext);
				bd.swapBW();
				bd.setInput(input);
				//bd.setInputNmlz(input);
				nn.learn(input, di.m_score, ALPHA);
			}
		}
		//cout << nn.dump() << "\n";
		cout << "RMS = " << calcRMS_RPO(nn, 10) << endl;
	}
}
double calcRMS_NNPO(HGNNet& nn, int N_GAME = 20)
{
	HGBoard bd;
	const int N_PO = 50;
	int nt = 0;		//	テスト数
	double sum2 = 0;
	vector<double> input(HG_NN_INSIZE);
	for (int i = 0; i < N_GAME; ++i) {
		cout << ".";
		bd.init();
		bool bt = true;
		for (int tcnt = 1;; ++tcnt, bt = !bt) {
			if( tcnt != 1 ) {
				if( bt ) {
					auto sc = bd.b_expctScoreNNPO(nn, N_PO);
					bd.setInputNmlz(input);
					auto ps = nn.predict(input);
					sum2 += (sc - ps) * (sc - ps);
				} else {
					HGBoard b2(bd.m_white, bd.m_black);		//	白黒反転盤面
					auto sc = b2.b_expctScoreNNPO(nn, N_PO);		//	[-3, +3]
					b2.setInputNmlz(input);
					auto ps = nn.predict(input);
					sum2 += (sc - ps) * (sc - ps);
				}
				++nt;
			}
			doRandomTurn(bd, bt, tcnt);
			if( bd.result() != 0 ) break;
		}
	}
	cout << "\n";
	return sqrt(sum2/nt);
}
void test_learnNNPO(bool verbose)
{
	const bool batchNrmlz = true;
	HGBoard bd;
	//
	HGNNet nn;
#if	0
	double ALPHA = 0.001;
#if	0
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " 1}, ReLU\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE}, RELU);
#elif 1
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " 1}, ReLU\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE}, RELU);
#else
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " 1}, ReLU\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE}, RELU);
#endif
#else
	double ALPHA = 0.01;
#if	0
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " 1}, tanh\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE}, TANH);
#elif 1
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " 1}, tanh\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE}, TANH, batchNrmlz);
#else
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " 1}, tanh\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE}, TANH);
#endif
#endif
	//nn.m_optSGD = true;
	vector<double> input(HG_NN_INSIZE);
	bd.setInputNmlz(input);
	auto sc = nn.predict(input);
	if( verbose )
		cout << "RMS = " << calcRMS_NNPO(nn) << endl;
	//
	vector<DataItem> data;
	readData(data, "data/NNPO100-001.txt");
	readData(data, "data/NNPO100-002.txt");
	readData(data, "data/NNPO100-003.txt");
	readData(data, "data/NNPO100-004.txt");
	readData(data, "data/NNPO100-005.txt");
	readData(data, "data/NNPO100-006.txt");
	readData(data, "data/NNPO100-007.txt");
	readData(data, "data/NNPO100-008.txt");
	readData(data, "data/NNPO100-009.txt");
	readData(data, "data/NNPO100-010.txt");
	if( data.empty() ) {
		cout << "can't open data file.\n";
		return;
	}
	for (int i = 0; i < 10; ++i) {
		double sum2 = 0;
		if( !verbose ) cout << "*";
		std::shuffle(data.begin(), data.end(), g_mt);
		for(const auto& di: data) {
			if( di.m_bturn ) {
				bd.set(di.m_ktext);
				//bd.setInput(input);
				bd.setInputNmlz(input);
				sum2 += nn.learn(input, di.m_score, ALPHA);
			} else {
				bd.set(di.m_ktext);
				bd.swapBW();
				//bd.setInput(input);
				bd.setInputNmlz(input);
				sum2 += nn.learn(input, di.m_score, ALPHA);
			}
		}
		if( verbose )
			cout << "RMS = " << calcRMS_NNPO(nn) << endl;
		else
			cout << "RMS = " << sqrt(sum2/data.size()) << endl;
	}
	if( !verbose )
		cout << "RMS = " << calcRMS_NNPO(nn) << endl;
	nn.save("NNPO1000x10.txt");
}
void test_learnRPO(bool verbose)
{
	const bool batchNrmlz = true;
	HGBoard bd;
	//
	HGNNet nn;
#if	0
	double ALPHA = 0.001;
#if	0
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " 1}, ReLU\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE}, RELU);
#elif 1
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " 1}, ReLU\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE}, RELU);
#else
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " 1}, ReLU\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE}, RELU);
#endif
#else
	double ALPHA = 0.01;
#if	0
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " 1}, tanh\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE}, TANH);
#elif 1
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " 1}, tanh\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE}, TANH, batchNrmlz);
#else
	cout << "# nodes of layers: {" << HG_NN_INSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " " << HG_NN_HIDSIZE << " 1}, tanh\n\n";
	nn.init(vector<int>{HG_NN_INSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE, HG_NN_HIDSIZE}, TANH);
#endif
#endif
	vector<double> input(HG_NN_INSIZE);
	bd.setInputNmlz(input);
	auto sc = nn.predict(input);
	if( verbose )
		cout << "RMS = " << calcRMS_RPO(nn) << endl;
	//
	vector<DataItem> data;
	readData(data, "data/RPO100-001.txt");
	readData(data, "data/RPO100-002.txt");
	readData(data, "data/RPO100-003.txt");
	readData(data, "data/RPO100-004.txt");
	readData(data, "data/RPO100-005.txt");
	readData(data, "data/RPO100-006.txt");
	readData(data, "data/RPO100-007.txt");
	readData(data, "data/RPO100-008.txt");
	readData(data, "data/RPO100-009.txt");
	readData(data, "data/RPO100-010.txt");
	if( data.empty() ) {
		cout << "can't open data file.\n";
		return;
	}
	for (int i = 0; i < 10; ++i) {
		if( !verbose ) cout << "*";
		std::shuffle(data.begin(), data.end(), g_mt);
		for(const auto& di: data) {
			if( di.m_bturn ) {
				bd.set(di.m_ktext);
				//bd.setInput(input);
				bd.setInputNmlz(input);
				nn.learn(input, di.m_score, ALPHA);
			} else {
				bd.set(di.m_ktext);
				bd.swapBW();
				//bd.setInput(input);
				bd.setInputNmlz(input);
				nn.learn(input, di.m_score, ALPHA);
			}
		}
		if( verbose )
			cout << "RMS = " << calcRMS_RPO(nn) << endl;
	}
	if( !verbose )
		cout << "RMS = " << calcRMS_RPO(nn) << endl;
	nn.save("RPO1000x10.txt");
}
void test_readData()
{
#ifdef _DEBUG
	const int N_GAME = 100;
#else
	const int N_GAME = 1000;
#endif
	HGBoard bd;
	ifstream ifs("data/RPO100-001.txt");
	string bw, ktext;
	double score;
	int cnt = 0;
	double sum2 = 0;
	int N_LOOP = 100;
#if	1
	vector<DataItem> data;
	while( !ifs.eof() ) {
		ifs >> bw;
		if( bw == "b" || bw == "w" ) {
			ifs >> ktext >> score;
			data.push_back(DataItem(bw == "b", ktext, score));
			++cnt;
		}
	}
	std::shuffle(data.begin(), data.end(), g_mt);
#else
	double sc2;
	while( cnt < N_LOOP && !ifs.eof() ) {
		ifs >> bw;
		if( bw == "b" || bw == "w" ) {
			ifs >> ktext >> score;
			bd.set(ktext);
			if( bw == "b" )
				sc2 = bd.b_expctScoreRPO(N_GAME);
			else
				sc2 = bd.w_expctScoreRPO(N_GAME);
			sum2 += (score - sc2) * (score - sc2);
			cout << bw << " " << ktext << " " << score << " " << sc2 << endl;
			++cnt;
		}
	}
	cout << "RMS = " << sqrt(sum2 / N_LOOP) << endl;
#endif
	cout << "OK\n";
}
double func(double x, double y) {    // x, y: [-1, +1]
    x *= 10;        //  [-10, +10]
    y *= 10;        //  [-10, +10]
    double R = sqrt(x*x + y*y) + 1e-4;
    return sin(R) / R;
}
void normalize(double& x, double& y) {  //  平均０，分散１ に正規化
    double ave = (x + y) / 2;
    double std2 = ((x-ave)*(x-ave) + (y-ave)*(y-ave)) / 2;
    x = (x - ave)/sqrt(std2 + 1e-4);
    y = (y - ave)/sqrt(std2 + 1e-4);
}
double func2RMS(HGNNet& nn, bool bNormalize, int N_LOOP = 1000)
{
    vector<data_t> input(2);
    double sum2 = 0;
    for(int i = 0; i != N_LOOP; ++i) {
        input[0] = g_rand11(g_mt);
        input[1] = g_rand11(g_mt);
        if( bNormalize )
	        normalize(input[0], input[1]);
        auto err = nn.predict(input) - func(input[0], input[1]);
        sum2 += err * err;
    }
    return sqrt(sum2/N_LOOP);
}
void test_2argsFunc()
{
	const bool bNormalize = false;
	cout << "f(x, y) = { R = sqrt(x*x + y*y) + 1e-4; return sin(R) / R; }\n";
	cout << "# node of layers: {2 20 20 1}, RELU," << (!bNormalize?"NOT":"") << " normalized:\n\n";
	cout << "N\tRMS\n";
	cout << "------- ----------\n";
    HGNNet nn;
    nn.init(vector<int>{2, 20, 20}, RELU, true, 0.001);
    //cout << nn.dump() << "\n";
    vector<data_t> input(2);
#if 0
    input[0] = g_rand11(g_mt);
    input[1] = g_rand11(g_mt);
    cout << input[0] << ", " << input[1] << " --> " << nn.predict(input) << "\n";
#endif
    //
    const int N_EPOCH = 1000000;
    for(int cnt = 0; cnt <= N_EPOCH; ++cnt) {
        input[0] = g_rand11(g_mt);      //  [-1, +1]
        input[1] = g_rand11(g_mt);      //  [-1, +1]
        if( bNormalize )
	        normalize(input[0], input[1]);
        nn.learn(input, func(input[0], input[1]));
		if( log10(cnt) == (int)log10(cnt) )
		{
			cout << "10^" << log10(cnt) << "\t" << func2RMS(nn, bNormalize) << endl;
		}
        
    }
    //
    cout << "\nOK\n";
}
double sinRMS(HGNNet& nn, int N_LOOP = 100)
{
	vector<double> input(1);
	double sum2 = 0;
	for (int i = 0; i < N_LOOP; ++i) {
		input[0] = g_rand11(g_mt);		//	[-1, +1]
		double err = nn.predict(input) - sin(input[0]*2*PI);
		sum2 += err * err;
	}
	return sqrt(sum2 / N_LOOP);
}
void test_12221()
{
	HGNNet nn;
	cout << "# node of layers: {1 2 2 2 1}, RELU:\n\n";
	nn.init(vector<int>{1, 2, 2, 2}, RELU);
	cout << nn.dump() << "\n";
	vector<double> input = {0.5};
	double T = 0.5;		//	教師値
	double y = nn.predict(input);
	double L = (y - T) * (y - T) / 2;
	cout << "y = " << y << ", L = " << L << "\n";
	//cout << nn.predict(input) << "\n";
	nn.calcError(input, 0.5);
	double dLdW = nn.m_layers[0][0].m_err * input[0];
	cout << "dLdW[0][0][0] = " << dLdW << endl;
	nn.m_layers[0][0].m_weight[0] += 0.01;
	cout << L << " + 0.01*" << dLdW << " = " << L + dLdW*0.01 << "\n";
	cout << nn.dump() << "\n";
	{
		double y = nn.predict(input);
		double L = (y - T) * (y - T) / 2;
		cout << "y = " << y << ", L = " << L << "\n";
	}
}
void test_120201()
{
	HGNNet nn;
#if	0
	cout << "# node of layers: {1 20 20 1}, tanh:\n\n";
	nn.init(vector<int>{1, 20, 20}, TANH);
#else
	cout << "# node of layers: {1 20 20 1}, RELU:\n\n";
	nn.init(vector<int>{1, 20, 20}, RELU);
#endif
	//nn.m_layers[0][0].m_weight[0] = 0.001;
	//cout << nn.dump() << "\n";
	vector<double> input = {0.5};
	double T = 0.5;		//	教師値
	double y = nn.predict(input);
	double L = (y - T) * (y - T) / 2;
	cout << "y = " << y << ", L = " << L << "\n";
	//cout << nn.predict(input) << "\n";
	nn.calcError(input, 0.5);
	double dLdW = nn.m_layers[0][0].m_err * input[0];
	cout << "dLdW[0][0][0] = " << dLdW << endl;
	nn.m_layers[0][0].m_weight[0] += 0.01;
	cout << L << " + 0.01*" << dLdW << " = " << L + dLdW*0.01 << "\n";
	//cout << nn.dump() << "\n";
	{
		double y = nn.predict(input);
		double L = (y - T) * (y - T) / 2;
		cout << "y = " << y << ", L = " << L << "\n";
	}
}
void test_1221()
{
	HGNNet nn;
	cout << "# node of layers: {1 2 2 1}, RELU:\n\n";
	nn.init(vector<int>{1, 2, 2}, RELU);
	cout << nn.dump() << "\n";
	vector<double> input = {0.5};
	double T = 0.5;		//	教師値
	double y = nn.predict(input);
	double L = (y - T) * (y - T) / 2;
	cout << "y = " << y << ", L = " << L << "\n";
	//cout << nn.predict(input) << "\n";
	nn.calcError(input, 0.5);
	double dLdW = nn.m_layers[0][0].m_err * input[0];
	cout << "dLdW[0][0][0] = " << dLdW << endl;
	nn.m_layers[0][0].m_weight[0] += 0.01;
	cout << L << " + 0.01*" << dLdW << " = " << L + dLdW*0.01 << "\n";
	cout << nn.dump() << "\n";
	{
		double y = nn.predict(input);
		double L = (y - T) * (y - T) / 2;
		cout << "y = " << y << ", L = " << L << "\n";
	}
}
void test_1201()
{
	HGNNet nn;
	cout << "# node of layers: {1 20 1}, RELU:\n\n";
	nn.init(vector<int>{1, 20}, RELU);
#if	0
	nn.m_layers[0][0].m_weight[0] = 1;
	nn.m_layers[0][0].m_weight[1] = 0;			//	バイアス
	nn.m_layers[0][1].m_weight[0] = -1;
	nn.m_layers[0][1].m_weight[1] = 0;			//	バイアス
	nn.m_layers[1][0].m_weight[0] = 0.9;
	nn.m_layers[1][0].m_weight[1] = 0.9;
	nn.m_layers[1][0].m_weight[2] = 0;			//	バイアス
#endif
	//cout << nn.dump() << "\n";
	vector<double> input = {0.5};
	double T = 0.5;		//	教師値
	double y = nn.predict(input);
	double L = (y - T) * (y - T) / 2;
	cout << "y = " << y << ", L = " << L << "\n";
	//cout << nn.predict(input) << "\n";
	nn.calcError(input, 0.5);
	double dLdW = nn.m_layers[0][0].m_err * input[0];
	cout << "dLdW[0][0][0] = " << dLdW << endl;
	nn.m_layers[0][0].m_weight[0] += 0.01;
	cout << L << " + 0.01*" << dLdW << " = " << L + dLdW*0.01 << "\n";
	//cout << nn.dump() << "\n";
	{
		double y = nn.predict(input);
		double L = (y - T) * (y - T) / 2;
		cout << "y = " << y << ", L = " << L << "\n";
	}
#if	0
	nn.train(input, 0.5);
	cout << nn.dumpBP() << "\n";
	cout << nn.dump() << "\n";
	cout << nn.predict(input) << "\n";
#endif
}
void test_121()
{
	HGNNet nn;
	cout << "# node of layers: {1 2 1}, RELU:\n\n";
	nn.init(vector<int>{1, 2}, RELU);
#if	0
	nn.m_layers[0][0].m_weight[0] = 1;
	nn.m_layers[0][0].m_weight[1] = 0;			//	バイアス
	nn.m_layers[0][1].m_weight[0] = -1;
	nn.m_layers[0][1].m_weight[1] = 0;			//	バイアス
	nn.m_layers[1][0].m_weight[0] = 0.9;
	nn.m_layers[1][0].m_weight[1] = 0.9;
	nn.m_layers[1][0].m_weight[2] = 0;			//	バイアス
#endif
	cout << nn.dump() << "\n";
	vector<double> input = {0.5};
	double T = 0.5;		//	教師値
	double y1 = nn.predict(input);
	double L1 = (y1 - T) * (y1 - T) / 2;
	cout << "y = " << y1 << ", L = " << L1 << "\n";
	//cout << nn.predict(input) << "\n";
	nn.calcError(input, 0.5);
	double dLdW = nn.m_layers[0][0].m_err * input[0];
	cout << "dLdW = " << dLdW << endl;
	nn.m_layers[0][0].m_weight[0] += 0.1;
	cout << nn.dump() << "\n";
	{
		double y1 = nn.predict(input);
		double L1 = (y1 - T) * (y1 - T) / 2;
		cout << "y = " << y1 << ", L = " << L1 << "\n";
	}
	//cout << nn.predict(input) << "\n";
#if	0
	nn.train(input, 0.5);
	cout << nn.dumpBP() << "\n";
	cout << nn.dump() << "\n";
	cout << nn.predict(input) << "\n";
#endif
}
void test_HGNNet()
{
	HGNNet nn;
	cout << "# node of layers: {1 50 50 1}, RELU:\n\n";
	cout << "N\tRMS\n";
	cout << "------- ----------\n";
	nn.init(vector<int>{1, 50, 50}, RELU);
	vector<double> input(1);
	//	学習
	for (int cnt = 1; cnt <= 1000000; ++cnt) {
		input[0] = g_rand11(g_mt);		//	[-1, +1]
		//double sc = nn.predict(input);		//	for test
		nn.learn(input, sin(input[0]*2*PI));
		//if( cnt == 10 || cnt == 100 || cnt == 1000 || cnt == 10000 || cnt == 100000 || cnt == 1000000 )
		if( log10(cnt) == (int)log10(cnt) )
		{
			cout << "10^" << log10(cnt) << "\t" << sinRMS(nn) << endl;
		}
	}
}
data_t linearArg1Func(data_t x) { return x*3 - 1; }
double linearArg1RMS(HGNNet& nn, int N_LOOP = 100)
{
	vector<double> input(1);
	double sum2 = 0;
	for (int i = 0; i < N_LOOP; ++i) {
		input[0] = g_rand11(g_mt);		//	[-1, +1]
		//normalize(input[0], input[1]);
		double err = nn.predict(input) - linearArg1Func(input[0]);
		sum2 += err * err;
	}
	return sqrt(sum2 / N_LOOP);
}
void test_linearFuncArg1()
{
	HGNNet nn;
	cout << "f(x) = 3*x - 1\n";
	double alpha = 10.0;
	cout << "# node of layers: {1 1}, alpha = " << alpha << "\n\n";
	nn.init(vector<int>{1}, SIGMOID, alpha);		//	1入力のみ（隠れ層無し）、活性化関数は無視される
	vector<double> input(1);
	//	学習・評価
	cout << "N\tRMS\n";
	cout << "------- ----------\n";
	for (int cnt = 1; cnt <= 10000; ++cnt) {
		input[0] = g_rand11(g_mt);		//	[-1, +1]
		//normalize(input[0], input[1]);
		nn.train(input, linearArg1Func(input[0]));
		if( log10(cnt) == (int)log10(cnt) )
		{
			cout << "10^" << log10(cnt) << "\t" << linearArg1RMS(nn) << "\t" << nn.dumpWeight(false);
		}
	}
	//cout << "\n" << nn.dump() << "\n";
}
data_t linearFunc(data_t x1, data_t x2) { return x1*3 - x2*2 + 1; }
double linearRMS(HGNNet& nn, int N_LOOP = 100)
{
	vector<double> input(2);
	double sum2 = 0;
	for (int i = 0; i < N_LOOP; ++i) {
		input[0] = g_rand11(g_mt);		//	[-1, +1]
		input[1] = g_rand11(g_mt);		//	[-1, +1]
		//normalize(input[0], input[1]);
		double err = nn.predict(input) - linearFunc(input[0], input[1]);
		sum2 += err * err;
	}
	return sqrt(sum2 / N_LOOP);
}
void test_linearFunc()
{
	HGNNet nn;
	cout << "f(x1, x2) = 3*x1 - 2*x2 + 1\n";
	cout << "# node of layers: {2 1}\n\n";
	nn.init(vector<int>{2}, SIGMOID, 0.01);		//	２入力のみ（隠れ層無し）、活性化関数は無視される
	nn.m_optSGD = true;
	vector<double> input(2);
	//	学習・評価
	cout << "N\tRMS\n";
	cout << "------- ----------\n";
	for (int cnt = 1; cnt <= 10000; ++cnt) {
		input[0] = g_rand11(g_mt);		//	[-1, +1]
		input[1] = g_rand11(g_mt);		//	[-1, +1]
		//normalize(input[0], input[1]);
		nn.train(input, linearFunc(input[0], input[1]));
		if( log10(cnt) == (int)log10(cnt) )
		{
			cout << "10^" << log10(cnt) << "\t" << linearRMS(nn) << "\t" << nn.dumpWeight(false);
		}
	}
	//cout << "\n" << nn.dump() << "\n";
}
#if	0
void test_linearFuncAF()
{
	HGNNet nn;
	cout << "f(x1, x2) = 3*x1 - 2*x2 + 1\n\n";
	vector<ActFunc> lst = {SIGMOID, TANH, RELU};
	for(auto af: lst) {
		cout << "# node of layers: {1 50 50 1}, ";
		switch( af ) {
		case SIGMOID:	cout << "SIGMOID:\n\n";	break;
		case TANH:		cout << "TANH:\n\n";	break;
		case RELU:		cout << "RELU:\n\n";	break;
		}
		cout << "# node of layers: {2 1}\n\n";
		nn.init(vector<int>{2}, af);		//	２入力のみ（隠れ層無し）
		vector<double> input(2);
		//	学習・評価
		cout << "N\tRMS\n";
		cout << "------- ----------\n";
		for (int cnt = 1; cnt <= 10000; ++cnt) {
			input[0] = g_rand11(g_mt);		//	[-1, +1]
			input[1] = g_rand11(g_mt);		//	[-1, +1]
			nn.train(input, linearFunc(input[0], input[1]));
			if( log10(cnt) == (int)log10(cnt) )
			{
				cout << "10^" << log10(cnt) << "\t" << linearRMS(nn) << "\t" << nn.dumpWeight(false);
			}
		}
	}
	//cout << "\n" << nn.dump() << "\n";
}
#endif
void test_sinFunc()
{
	HGNNet nn;
	vector<ActFunc> lst = {SIGMOID, TANH, RELU};
	for(auto af: lst) {
		cout << "# node of layers: {1 50 50 1}, ";
		switch( af ) {
		case SIGMOID:	cout << "SIGMOID:\n\n";	break;
		case TANH:		cout << "TANH:\n\n";	break;
		case RELU:		cout << "RELU:\n\n";	break;
		}
		cout << "N\tRMS\n";
		cout << "------- ----------\n";
		nn.init(vector<int>{1, 50, 50}, af);
		vector<double> input(1);
		//	学習
		for (int cnt = 1; cnt <= 100000; ++cnt) {
			input[0] = g_rand11(g_mt);		//	[-1, +1]
			//double sc = nn.predict(input);		//	for test
			nn.train(input, sin(input[0]*2*PI));
			//if( cnt == 10 || cnt == 100 || cnt == 1000 || cnt == 10000 || cnt == 100000 || cnt == 1000000 )
			if( log10(cnt) == (int)log10(cnt) )
			{
				cout << "10^" << log10(cnt) << "\t" << sinRMS(nn) << endl;
			}
		}
		cout << endl;
	}
}
#if	0
//	ランダムプレイアウトにより 状態 → 期待スコア学習データ生成
void otg_genDataRPO()
{
#ifdef _DEBUG
	const int N_GAME = 100;
#else
	const int N_GAME = 1000;
#endif
	OTGBoard bd;
	int d1, d2;
	Moves mvs;
	for (int g = 1; g <= 100; ++g) {
		cout << "#" << g << endl;
		bd.init();
		//cout << bd.text() << endl;
		bool bt = true;
		for (int cnt = 1;; ++cnt, bt = !bt) {
			if( cnt != 1 ) {
				if( bt ) {
					cout << "b " << bd.ktext() << " " << bd.b_expctScoreRPO(N_GAME) << endl;
					//cout << "black turn exp score = " << bd.b_expctScoreRPO(N_GAME) << endl << endl;
				} else {
					cout << "w " << bd.ktext() << " " << bd.w_expctScoreRPO(N_GAME) << endl;
					//cout << "white turn exp score = " << bd.w_expctScoreRPO(N_GAME) << endl << endl;
				}
			}
			otg_doRandomTurn(bd, bt, cnt, d1, d2, mvs);
			//for (const auto& mv : mvs) cout << mv.text() << " ";
			//cout << "\n";
			//cout << bd.text();
			//cout << bd.ktext() << endl;
			if (bd.result() != 0) break;
		}
	}
}
#endif
//	ランダムプレイアウトにより 状態 → 期待スコア学習データ生成
void genDataRPO()
{
#ifdef _DEBUG
	const int N_GAME = 100;
#else
	const int N_GAME = 1000;
#endif
	HGBoard bd;
	for (int g = 1; g <= 100; ++g) {
		cout << "# " << g << endl;
		bd.init();
		//cout << bd.text() << endl;
		bool bt = true;
		for (int cnt = 1;; ++cnt, bt = !bt) {
			if( cnt != 1 ) {
				if( bt ) {
					cout << "b " << bd.ktext() << " " << bd.b_expctScoreRPO(N_GAME) << endl;
					//cout << "black turn exp score = " << bd.b_expctScoreRPO(N_GAME) << endl << endl;
				} else {
					cout << "w " << bd.ktext() << " " << bd.w_expctScoreRPO(N_GAME) << endl;
					//cout << "white turn exp score = " << bd.w_expctScoreRPO(N_GAME) << endl << endl;
				}
			}
			int d1, d2;
			do {
				d1 = g_mt() % 3 + 1;
				d2 = g_mt() % 3 + 1;
			} while (d1 == d2 && cnt == 1);
			//cout << cnt << ") " << (bt?"black ":"white ") << d1 << d2 << ": ";
			MovesList lst;
			Moves mvs;
			if (bt) {
				bd.b_genMovesList(lst, d1, d2);
				if (!lst.empty()) {
					mvs = lst[g_mt() % lst.size()];
					bd.b_move(mvs);
				}
			}
			else {
				bd.w_genMovesList(lst, d1, d2);
				if (!lst.empty()) {
					mvs = lst[g_mt() % lst.size()];
					bd.w_move(mvs);
				}
			}
			//for (const auto& mv : mvs) cout << mv.text() << " ";
			//cout << "\n";
			//cout << bd.text();
			//cout << bd.ktext() << endl;
			if (bd.result() != 0) break;
		}
	}
}
//	学習済みプレイアウトにより 状態 → 期待スコア学習データ生成
void genDataNN()
{
	HGNNet nn;
	bool rc = nn.load("RPO1000x10.txt");
	assert( rc );
	//	局面期待値スコア計算のためのゲーム数
#ifdef _DEBUG
	const int N_GAME = 10;
#else
	const int N_GAME = 100;
#endif
	HGBoard bd;
	for (int g = 1; g <= 100; ++g) {		//	100対局
		cout << "# " << g << endl;
		bd.init();
		//cout << bd.text() << endl;
		bool bt = true;
		for (int cnt = 1;; ++cnt, bt = !bt) {
			if( cnt != 1 ) {
				if( bt ) {
					cout << "b " << bd.ktext() << " " << bd.b_expctScoreNNPO(nn, N_GAME) << endl;
					//cout << "black turn exp score = " << bd.b_expctScoreRPO(N_GAME) << endl << endl;
				} else {
					cout << "w " << bd.ktext() << " " << bd.w_expctScoreNNPO(nn, N_GAME) << endl;
					//cout << "white turn exp score = " << bd.w_expctScoreRPO(N_GAME) << endl << endl;
				}
			}
			int d1, d2;
			do {
				d1 = g_mt() % 3 + 1;
				d2 = g_mt() % 3 + 1;
			} while (d1 == d2 && cnt == 1);
			//cout << cnt << ") " << (bt?"black ":"white ") << d1 << d2 << ": ";
			MovesList lst;
			Moves mvs;
			if (bt) {
				bd.b_genMovesList(lst, d1, d2);
				if (!lst.empty()) {
					mvs = lst[g_mt() % lst.size()];
					bd.b_move(mvs);
				}
			}
			else {
				bd.w_genMovesList(lst, d1, d2);
				if (!lst.empty()) {
					mvs = lst[g_mt() % lst.size()];
					bd.w_move(mvs);
				}
			}
			//for (const auto& mv : mvs) cout << mv.text() << " ";
			//cout << "\n";
			//cout << bd.text();
			//cout << bd.ktext() << endl;
			if (bd.result() != 0) break;
		}
	}
}
void test_expctScoreRPO2()
{
	HGBoard bd;
	//bd.set(	"6200000000000003300002000000");
	bd.set(	"6002000000000062000000000000");
	//bd.set(	"6002000000000060200000000000");
	//bd.set(	"6020000000000060200000000000");		//	5/9 + 4/9(4/9 - 5/9) = (45 - 1)/81 = 0.543
	//bd.set(	"6020000000000062000000000000");		//	5/9 - 4/9 = 0.111…
	//bd.set(		"6020000000000071000000000000");		//	5/9 - 4/9 = 0.111…
	cout << bd.text() << endl;
#ifdef _DEBUG
	const int N_GAME = 100;
#else
	const int N_GAME = 1000;
#endif
	cout << "black turn exp score = " << bd.b_expctScoreRPO(N_GAME) << endl;
	cout << "white turn exp score = " << bd.w_expctScoreRPO(N_GAME) << endl;
}
void test_expctScoreRPO()
{
	HGBoard bd;
#if	1
	cout << bd.text() << endl;
	bool bt = true;
	for (int cnt = 1;; ++cnt, bt = !bt) {
		if( cnt != 1 ) {
#ifdef _DEBUG
	const int N_GAME = 100;
#else
	const int N_GAME = 1000;
#endif
			if( bt ) {
				cout << "black turn exp score = " << bd.b_expctScoreRPO(N_GAME) << endl << endl;
			} else {
				cout << "white turn exp score = " << bd.w_expctScoreRPO(N_GAME) << endl << endl;
			}
		}
		int d1, d2;
		do {
			d1 = g_mt() % 3 + 1;
			d2 = g_mt() % 3 + 1;
		} while (d1 == d2 && cnt == 1);
		cout << cnt << ") " << (bt?"black ":"white ") << d1 << d2 << ": ";
		MovesList lst;
		Moves mvs;
		if (bt) {
			bd.b_genMovesList(lst, d1, d2);
			if (!lst.empty()) {
				mvs = lst[g_mt() % lst.size()];
				bd.b_move(mvs);
			}
		}
		else {
			bd.w_genMovesList(lst, d1, d2);
			if (!lst.empty()) {
				mvs = lst[g_mt() % lst.size()];
				bd.w_move(mvs);
			}
		}
		for (const auto& mv : mvs) cout << mv.text() << " ";
		cout << "\n";
		cout << bd.text();
		cout << bd.ktext() << endl;
		if (bd.result() != 0) break;
	}
#else
	bd.clear();
	bd.b_setAt(3, 2);
	bd.w_setAt(3, 2);
	cout << bd.text() << endl;
	cout << bd.b_expctScoreRPO(100) << endl;
#endif
}
void test_randomPlayOut()
{
	HGBoard bd;
	bool bt = true;
	for (int cnt = 1; !bd.result(); ++cnt, bt = !bt) {
		cout << bd.text() << "\n";
		int d1, d2;
		do {
			d1 = g_mt() % 3 + 1;
			d2 = g_mt() % 3 + 1;
		} while (d1 == d2 && cnt == 1);
		cout << cnt << ") " << d1 << d2 << ": ";
		MovesList lst;
		Moves mvs;
		if (bt) {
			bd.b_genMovesList(lst, d1, d2);
			if (!lst.empty()) {
				mvs = lst[g_mt() % lst.size()];
				bd.b_move(mvs);
			}
		}
		else {
			bd.w_genMovesList(lst, d1, d2);
			if (!lst.empty()) {
				mvs = lst[g_mt() % lst.size()];
				bd.w_move(mvs);
			}
		}
		for (const auto& mv : mvs) cout << mv.text() << " ";
		cout << "\n";
		if (bd.result() != 0) break;
		//if (cnt == 100) break;
	}
	cout << bd.text() << "\n";
	cout << "resultSGB = " << bd.resultSGB() << endl;
}
void test_genMoves()
{
	HGBoard bd;
	bd.set("0100000004210004000220000000");
	cout << bd.text() << endl;
	MovesList lst;
	bd.b_genMovesList(lst, 2, 1);
	for(auto& mvs: lst) {
		for(auto& mv: mvs) cout << mv.text() << " ";
		cout << endl;
	}
}
void test_load_save()
{
	HGNNet nn, nn2;
	nn.init(vector<int>{2, 10, 10}, TANH);
	auto r = nn.save("dump.txt");
	assert( r );
	r = nn2.load("dump.txt");
	assert( r );
	assert( nn == nn2 );
}
void test_negaMax1_prime()
{
	HGNNet nn;
	bool rc = nn.load("RPO1000x10.txt");
	assert( rc );
	HGBoard bd;
	bd.set("0022002110000002020000100030");
	cout << bd.text() << "\n";
	cout << "black expct score = " << bd.b_expctScore(nn) << "\n";
	Moves mvs;
	bd.negaMax1(mvs, nn, 1, 1);
	cout << "11: ";
	for(auto mv: mvs) cout << mv.text() << " ";
	cout << "\n";
}
void test_negaMax1()
{
	HGNNet nn;
	bool rc = nn.load("RPO1000x10.txt");
	assert( rc );
	HGBoard bd;
	cout << bd.text() << "\n";
	bool bt = true;
	for(int cnt = 1; ; ++cnt, bt = !bt) {		//	終局でない間
		int d1, d2;
		do {
			d1 = g_mt() % 3 + 1;
			d2 = g_mt() % 3 + 1;
		} while (d1 == d2 && cnt == 1);
		Moves mvs;
		if( bt ) {
			bd.negaMax1(mvs, nn, d1, d2);
			if( !mvs.empty() )
				bd.b_move(mvs);
		} else {
			HGBoard b2(bd.m_white, bd.m_black);		//	白黒反転
			b2.negaMax1(mvs, nn, d1, d2);
			if( !mvs.empty() )
				bd.w_move(mvs);
		}
		cout << cnt << ") " << (bt?"black ":"white ") << d1 << d2 << ": ";
		for(auto mv: mvs) cout << mv.text() << " ";
		cout << "\n";
		cout << bd.text();	// << "\n";
		if( bd.result() != 0 ) break;
		if( !bt ) {
			cout << "black: exp score = " << bd.b_expctScore(nn) << "\n\n";
		} else {
			cout << "white: exp score = " << bd.w_expctScore(nn) << "\n\n";
		}
	}
}
void test_negaMax1_random()
{
	HGNNet nn;
	bool rc = nn.load("RPO1000x10.txt");
	assert( rc );
	HGBoard bd;
	cout << bd.text() << "\n";
	bool bt = true;
	for(int cnt = 1; ; ++cnt, bt = !bt) {		//	終局でない間
		int d1, d2;
		do {
			d1 = g_mt() % 3 + 1;
			d2 = g_mt() % 3 + 1;
		} while (d1 == d2 && cnt == 1);
		Moves mvs;
		if( bt ) {
			bd.negaMax1(mvs, nn, d1, d2);
			if( !mvs.empty() )
				bd.b_move(mvs);
		} else {
			MovesList lst;
			bd.w_genMovesList(lst, d1, d2);
			if (!lst.empty()) {
				mvs = lst[g_mt() % lst.size()];
				bd.w_move(mvs);
			}
		}
		cout << cnt << ") " << (bt?"black ":"white ") << d1 << d2 << ": ";
		for(auto mv: mvs) cout << mv.text() << " ";
		cout << "\n";
		cout << bd.text();	// << "\n";
		if( bd.result() != 0 ) break;
		if( !bt ) {
			cout << "black: exp score = " << bd.b_expctScore(nn) << "\n\n";
		} else {
			cout << "white: exp score = " << bd.w_expctScore(nn) << "\n\n";
		}
	}
}
void test_negaMax1_random_stat(int N_GAME)
{
	cout << "Black(NN) vs White(Random):\n";
	HGNNet nn;
	//bool rc = nn.load("RPO1000x10.txt");
	bool rc = nn.load("NNPO1000x10.txt");
	assert( rc );
	HGBoard bd;
	int nBlackWin = 0;
	for (int gc = 0; gc != N_GAME; ++gc) {
		if( gc % 10 == 0 ) cout << ".";
		bd.init();
		//cout << bd.text() << "\n";
		bool bt = true;
		for(int cnt = 1; ; ++cnt, bt = !bt) {		//	終局でない間
			int d1, d2;
			do {
				d1 = g_mt() % 3 + 1;
				d2 = g_mt() % 3 + 1;
			} while (d1 == d2 && cnt == 1);
			Moves mvs;
			if( bt ) {
				bd.negaMax1(mvs, nn, d1, d2);
				if( !mvs.empty() )
					bd.b_move(mvs);
			} else {
				MovesList lst;
				bd.w_genMovesList(lst, d1, d2);
				if (!lst.empty()) {
					mvs = lst[g_mt() % lst.size()];
					bd.w_move(mvs);
				}
			}
			//cout << cnt << ") " << (bt?"black ":"white ") << d1 << d2 << ": ";
			//for(auto mv: mvs) cout << mv.text() << " ";
			//cout << "\n";
			//cout << bd.text();	// << "\n";
			if( bd.result() != 0 ) break;
			if( !bt ) {
				//cout << "black: exp score = " << bd.b_expctScore(nn) << "\n\n";
			} else {
				//cout << "white: exp score = " << bd.w_expctScore(nn) << "\n\n";
			}
		}
		if( bd.result() > 0 ) nBlackWin += 1;
	}
	cout << "\n";
	cout << "Black win rate: " << nBlackWin << " / " << N_GAME << " = " << nBlackWin*100.0/N_GAME << "%\n";
}
