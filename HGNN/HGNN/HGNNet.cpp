//----------------------------------------------------------------------
//
//		ハーフギャモン向けニューラルネットワーク
//		HGNNet クラス実装ファイル
//		Copyright (C) 2020 by N.Tsuda
//		License: MIT
//
//----------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <random>
#include <assert.h>
#include "HGNNet.h"
#include "utils.h"

using namespace std;

#if	0
static std::random_device	g_rd;
static std::mt19937	g_mt;
static std::uniform_real_distribution<> g_rand11(-1, +1);        // [-1, +1] 範囲の一様乱数
#endif

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double tanhDiff(double x) {
	auto ch = cosh(x);
	return 1.0 / (ch * ch);
}
const double LR_DIFF = 0.01;		//	Leaky-ReLU(x) x<0 の場合の傾き 
//----------------------------------------------------------------------
HGNNet::HGNNet()
//: m_actFunc(NONE)
//, m_alpha(0.01)
//, m_nInput(0)
{
	m_actFunc = NONE;
	m_alpha = 0.01;
	m_nInput = 0;
}
void HGNNet::init(const std::vector<int>& lst, ActFunc actFunc, bool batchNrmlz, double alpha)
{
	init(lst, actFunc, alpha);
	m_batchNrmlz = batchNrmlz;
}
void HGNNet::init(const std::vector<int>& lst, ActFunc actFunc, OutputType ot, double alpha)
{
	init(lst, actFunc, alpha);
	m_outputType = ot;
}
//	第１引数：入力・隠れ層レイヤーのノード数リスト、出力レイヤーは指定しない
//	第２引数：活性化関数種別
void HGNNet::init(const std::vector<int>& lst, ActFunc actFunc, double alpha)
{
	if (lst.empty()) {
		return;
	}
	m_optSGD = false;
	m_batchNrmlz = true;
	m_outputType = OT_REGRESSION;
	m_actFunc = actFunc;
	m_alpha = alpha;
	m_nInput = lst[0];		//	入力層ノード数
	m_layers.resize(lst.size());
	//	隠れ層初期化
	for (int l = 1; l != lst.size(); ++l) {
		auto& layer = m_layers[l - 1];
		layer.resize(lst[l]);
		for (int n = 0; n != lst[l]; ++n) {
			layer[n].m_weight.resize(lst[l - 1] + 1);		//	+1 for バイアス
			for (auto& w : layer[n].m_weight) w = g_rand11(g_mt);		//	係数 [-1, +1] 乱数初期化
			layer[n].m_wtDiff.resize(lst[l - 1] + 1);		//	+1 for バイアス
			for (auto& w : layer[n].m_wtDiff) w = 0;		//	
		}
	}
	//	出力層初期化
	int l = lst.size() - 1;
	auto& layer = m_layers[l];		//	
	layer.resize(1);
	if (l != 0) {	//	隠れ層がある場合
		layer[0].m_weight.resize(lst[l] + 1);		//	+1 for バイアス
		layer[0].m_wtDiff.resize(lst[l] + 1);		//	+1 for バイアス
	} else {			//	隠れ層が無い場合
		layer[0].m_weight.resize(lst[0] + 1);		//	+1 for バイアス
		layer[0].m_wtDiff.resize(lst[0] + 1);		//	+1 for バイアス
	}
	for (auto& w : layer[0].m_weight) w = g_rand11(g_mt);		//	係数 [-1, +1] 乱数初期化
	for (auto& w : layer[0].m_wtDiff) w = 0;
}
//	入力 → 回帰予測
data_t HGNNet::predict(const std::vector<data_t>& input)			//	予測
{
	if (input.size() != m_nInput) {
		//cout << "input.size() = " << input.size() << ", m_nInput = " << m_nInput << "\n";
		assert(0);
		return 0;
	}
	for (int l = 0; l != m_layers.size(); ++l) {		//	for 各レイヤー
		const bool outLayer = l == m_layers.size() - 1;
		vector<HGNNNode>& layer = m_layers[l];
		double sum = 0, sum2 = 0;		//	for バッチ・ノーマライゼーション
		for (int n = 0; n != layer.size(); ++n) {		//	レイヤーの全ノードを処理
			HGNNNode& node = layer[n];
			node.m_sum = node.m_weight.back();		//	バイアス
			if (l == 0) {		//	前段が入力層の場合
				for (int i = 0; i != input.size(); ++i) {
					node.m_sum += input[i] * node.m_weight[i];
				}
			} else {
				for (int i = 0; i != node.m_weight.size() - 1; ++i) {
					node.m_sum += m_layers[l - 1][i].m_output * node.m_weight[i];
				}
			}
			switch ( !outLayer ? m_actFunc : NONE) {
			case SIGMOID:
				node.m_output = sigmoid(node.m_sum);
				node.m_diff = (1 - node.m_output) * node.m_output;		//	微分値
				assert( !isnan(node.m_output) );
				break;
			case TANH:
				node.m_output = tanh(node.m_sum);
				node.m_diff = tanhDiff(node.m_sum);		//	微分値
				assert( !isnan(node.m_output) );
				break;
			case RELU:
				//node.m_output = std::max(0.0, node.m_sum);
				node.m_output = node.m_sum >= 0 ? node.m_sum : 0;
				node.m_diff = node.m_sum >= 0 ? 1 : 0;		//	微分値
				assert( !isnan(node.m_output) );
				break;
			case LEAKY_RELU:
				node.m_output = node.m_sum >= 0 ? node.m_sum : node.m_sum * LR_DIFF;
				node.m_diff = node.m_sum >= 0 ? 1 : LR_DIFF;		//	微分値
				assert( !isnan(node.m_output) );
				break;
			default:
				node.m_output = node.m_sum;		//	恒等関数
				node.m_diff = 1.0;			//	f(x) = x の微分は 1.0
				assert( !isnan(node.m_output) );
				break;
			}
			if( !outLayer && m_batchNrmlz ) {
				sum += node.m_output;
				sum2 += node.m_output * node.m_output;
			}
		}
		if( !outLayer && m_batchNrmlz ) {
			double ave = sum / layer.size();
			double std2 = sum2 / layer.size() - ave * ave;
			double denom = 1.0 / sqrt(std2 + 1e-6);
			for (int n = 0; n != layer.size(); ++n) {		//	レイヤーの全ノードを処理
				HGNNNode& node = layer[n];
				node.m_output = (node.m_output - ave) * denom;
			}
		}
	}
	if( m_outputType == OT_TANH)
		return tanh(m_layers.back()[0].m_output);
	else
		return m_layers.back()[0].m_output;
}
//	学習、第１引数：入力値、第２引数：教師値
void HGNNet::learn(const std::vector<data_t>& input, data_t t, double alpha)
{
	const double ETA = 0.9;
	if( alpha > 0 ) m_alpha = alpha;
#if	1
	calcError(input, t);
	for (int l = m_layers.size(); --l >= 0;) {		//	全レイヤーについて
		HGNNLayer& layer = m_layers[l];
		for (int n = 0; n != layer.size(); ++n) {
			HGNNNode& node = layer[n];
			if (l > 0) {		//	前段が隠れ層の場合
				auto& prevLayer = m_layers[l - 1];
				for (int k = 0; k != prevLayer.size(); ++k) {		//	前段の全ノードについて
					auto& prevNode = prevLayer[k];
					if( m_optSGD ) {
						auto t = node.m_wtDiff[k] * ETA;
						node.m_weight[k] -= t + (node.m_wtDiff[k] = m_alpha * node.m_err * prevLayer[k].m_output);	//	重み修正
					} else
						node.m_weight[k] -= m_alpha * node.m_err * prevLayer[k].m_output;	//	重み修正
					assert( !isnan(node.m_weight[k]) );
					assert( abs(node.m_weight[k]) < 1e12 );
				}
			} else {			//	前段が入力層の場合
				for (int k = 0; k != node.m_weight.size() - 1; ++k) {		//	-重み係数配列の最後はバイアス用のために -1
					if( m_optSGD ) {
						auto t = node.m_wtDiff[k] * ETA;
						node.m_weight[k] -= t + (node.m_wtDiff[k] = m_alpha * node.m_err * input[k]);	//	重み修正
					} else
						node.m_weight[k] -= m_alpha * node.m_err * input[k];
					assert( !isnan(node.m_weight[k]) );
					assert( abs(node.m_weight[k]) < 1e12 );
				}
			}
			if( m_optSGD ) {
				auto t = node.m_wtDiff.back() * ETA;
				node.m_weight.back() -= t + (node.m_wtDiff.back() = m_alpha * node.m_err);	//	重み修正
			} else
				node.m_weight.back() -= m_alpha * node.m_err;	//	for バイアス
			assert( !isnan(node.m_weight.back()) );
			assert( abs(node.m_weight.back()) < 1e12 );
		}
	}
#else
	for (auto& layer : m_layers) {
		for (auto& node : layer)
			node.m_err = 0;			//	すべての誤差をいったんクリア
	}
	//
	data_t y = predict(input);
	//m_layers.back().front().m_err = y - t;			//	出力ノード誤差
	for (int l = m_layers.size(); --l >= 0;) {		//	全レイヤーについて
		HGNNLayer& layer = m_layers[l];
		for (int n = 0; n != layer.size(); ++n) {
			HGNNNode& node = layer[n];
			if (l == m_layers.size() - 1) {	//	出力層の場合
				if( m_outputType == OT_TANH )
					node.m_err = y - log((1+t)/(1-t)) / 2;			//	∂L/∂y、L = (y-t)^2/2
				else
					node.m_err = y - t;			//	∂L/∂y、L = (y-t)^2/2
			}
			if (l > 0) {		//	前段が隠れ層の場合
				auto& prevLayer = m_layers[l - 1];
				//for (int i = 0; i != node.m_weight.size() - 1; ++i) {		//	-重み係数配列の最後はバイアス用のために -1
				for (int k = 0; k != prevLayer.size(); ++k) {		//	前段の全ノードについて
					auto& prevNode = prevLayer[k];
					//cout	<< "l, n, k = " << l << ", " << n << ", " << k << ":\n";
					//cout	<< "  prevNode.m_err = " << prevNode.m_err
					//		<< ", prevNode.m_diff = " << prevNode.m_diff << "\n";
					//cout	<< "  node.m_err = " << node.m_err
					//		<< ", node.m_weight = " << node.m_weight[k]<< "\n";
					prevNode.m_err += prevNode.m_diff * node.m_err * node.m_weight[k];		//	誤差逆伝搬
					//cout	<< "  --> " << prevNode.m_err << "\n";
					node.m_weight[k] -= m_alpha * node.m_err * prevLayer[k].m_output;	//	重み修正
				}
				//node.m_weight[i] -= m_alpha * node.m_err * m_layers[l-1][i].m_output;	//	重み修正
			//}
			}
			else {			//	前段が入力層の場合
				for (int i = 0; i != node.m_weight.size() - 1; ++i) {		//	-重み係数配列の最後はバイアス用のために -1
					node.m_weight[i] -= m_alpha * node.m_err * input[i];
				}
			}
			node.m_weight.back() -= m_alpha * node.m_err;	//	for バイアス
		}
	}
#endif
}
//	誤差逆伝搬計算のみ for Test
void HGNNet::calcError(const std::vector<data_t>& input, data_t t)
{
	for (auto& layer : m_layers) {
		for (auto& node : layer)
			node.m_err = 0;			//	すべての誤差をいったんクリア
	}
	//
	data_t y = predict(input);
	//m_layers.back().front().m_err = y - t;			//	出力ノード誤差
	for (int l = m_layers.size(); --l >= 0;) {		//	全レイヤーについて
		HGNNLayer& layer = m_layers[l];
		for (int n = 0; n != layer.size(); ++n) {
			HGNNNode& node = layer[n];
			assert( abs(node.m_err) < 1e12 );
			if (l == m_layers.size() - 1) {	//	出力層の場合
				if( m_outputType == OT_TANH )
					node.m_err = y - log((1+t)/(1-t)) / 2;			//	∂L/∂y、L = (y-t)^2/2
				else
					node.m_err = y - t;			//	∂L/∂y、L = (y-t)^2/2
				assert( abs(node.m_err) < 1e12 );
			}
			if (l > 0) {		//	前段が隠れ層の場合
				auto& prevLayer = m_layers[l - 1];
				//for (int i = 0; i != node.m_weight.size() - 1; ++i) {		//	-重み係数配列の最後はバイアス用のために -1
				for (int k = 0; k != prevLayer.size(); ++k) {		//	前段の全ノードについて
					auto& prevNode = prevLayer[k];
					//cout	<< "l, n, k = " << l << ", " << n << ", " << k << ":\n";
					//cout	<< "  prevNode.m_err = " << prevNode.m_err
					//		<< ", prevNode.m_diff = " << prevNode.m_diff << "\n";
					//cout	<< "  node.m_err = " << node.m_err
					//		<< ", node.m_weight = " << node.m_weight[k]<< "\n";
					prevNode.m_err += prevNode.m_diff * node.m_err * node.m_weight[k];		//	誤差逆伝搬
					//cout	<< "  --> " << prevNode.m_err << "\n";
					//node.m_weight[k] -= m_alpha * node.m_err * prevLayer[k].m_output;	//	重み修正
					assert( abs(prevNode.m_err) < 1e12 );
				}
				//node.m_weight[i] -= m_alpha * node.m_err * m_layers[l-1][i].m_output;	//	重み修正
			//}
			} else {			//	前段が入力層の場合
				//for (int i = 0; i != node.m_weight.size() - 1; ++i) {		//	-重み係数配列の最後はバイアス用のために -1
				//	node.m_weight[i] -= m_alpha * node.m_err * input[i];
				//}
			}
			//node.m_weight.back() -= m_alpha * node.m_err;	//	for バイアス
		}
	}
}
bool HGNNet::operator==(const HGNNet& nn) const
{
	if( m_layers.size() != nn.m_layers.size() ) return false;
	for(int l = 0; l != m_layers.size(); ++l) {
		if( m_layers[l].size() != nn.m_layers[l].size() )
			return false;
		for (int n = 0; n != m_layers[l].size(); ++n) {
			if( m_layers[l][n].m_weight.size() != nn.m_layers[l][n].m_weight.size() )
				return false;
			for (int i = 0; i != m_layers[l][n].m_weight.size(); ++i) {
				if( (float)m_layers[l][n].m_weight[i] != (float)nn.m_layers[l][n].m_weight[i] )
					return false;
			}
		}
	}
	return true;
}
std::string HGNNet::dumpWeight(bool highPrec) const		//	重み係数のみ
{
	string txt;
	//stringstream ss;
	char buf[30];		//	for double text
	//string buf;	buf.resize(30);		//	for double text
	for (const auto& layer : m_layers) {
		txt += "wt: ";
		for (const auto& node : layer) {
			txt += "( ";
			for (auto w : node.m_weight) {
				//txt += to_string(w) + " ";
				//ss << w;
				//txt += ss.str() + " ";
				if( highPrec ) {
					sprintf_s(buf, "%.17g", w);
					txt += string(buf) + " ";
				} else
					txt += to_string(w) + " ";
			}
			txt += ") ";
		}
		txt += "\n";
	}
	return txt;
}
std::string HGNNet::dump() const
{
	string txt = "#node of layers: ";
	txt += to_string(m_nInput) + " ";
	for (const auto& layer : m_layers)
		txt += to_string(layer.size()) + " ";
	txt += "\n";
	txt += "actFunc: ";
	switch( m_actFunc ) {
	case SIGMOID:	txt += "SIGMOID";	break;
	case TANH:	txt += "TANH";	break;
	case RELU:	txt += "RELU";	break;
	}
	txt += "\n";
	txt += dumpWeight();
	return txt;
}
void HGNNet::makeWeightSeq()			//	係数を 0.1, 0.2, ... に設定、for Test
{
	for (auto& layer : m_layers) {
		for (auto& node : layer) {
			data_t d = 0;
			for (auto& w : node.m_weight)
				w = d += 0.1;
		}
	}
}
std::string HGNNet::dumpPredict(const std::vector<data_t>& input) //const
{
	string txt = "input: ";
	predict(input);
	for (auto x : input) txt += to_string(x) + " ";
	txt += "\n";
	for (const auto& layer : m_layers) {
		txt += "(sum, out) = ";
		for (const auto& node : layer) {
			txt += "(" + to_string(node.m_sum) + ", " + to_string(node.m_output) + ") ";
		}
		txt += "\n";
	}
	return txt;
}
//	for Test、バックプロパゲーションのための誤差・微分値を表示
std::string HGNNet::dumpBP() const
{
	string txt;
	for (const auto& layer : m_layers) {
		txt += "(err, diff) = ";
		for (const auto& node : layer) {
			txt += "(" + to_string(node.m_err) + ", " + to_string(node.m_diff) + ") ";
		}
		txt += "\n";
	}
	return txt;
}
bool HGNNet::save(cchar* fname) const		//	指定ファイルにネットワークの状態を保存
{
	ofstream ofs(fname);
	if( !ofs ) return false;
	ofs << dump();
	ofs.close();
	return true;
}
bool HGNNet::load(cchar* fname)
{
	ifstream ifs(fname);
	if( !ifs ) return false;
	string buf, buf2, buf3;
	ifs >> buf >> buf2 >> buf3;
	if( buf != "#node" || buf2 != "of" || buf3 != "layers:" ) {
		ifs.close();
		return false;
	}
	vector<int> lst;
	for (;;) {
		ifs >> buf;
		if( buf.empty() || !isdigit(buf[0]) ) break;
		lst.push_back(atoi(&buf[0]));
	}
	if( lst.back() != 1 ) return false;		//	出力層は回帰のみサポート
	lst.pop_back();		//	remove 出力層
	ActFunc af;
	if( buf != "actFunc:" ) return false;
	ifs >> buf;
	if( buf == "SIGMOID" ) af = SIGMOID;
	else if( buf == "TANH" ) af = TANH;
	else if( buf == "RELU" ) af = RELU;
	else return false;
	init(lst, af);
	//
	ifs >> buf;
	for(int l = 0; buf == "wt:"; ++l) {
		if( l >= m_layers.size() ) return false;
		ifs >> buf;
		for(int n = 0; buf == "("; ++n) {
			if( n >= m_layers[l].size() ) return false;
			for (int i = 0;; ++i) {
				ifs >> buf;
				if (buf == ")") break;
				if( i >= m_layers[l][n].m_weight.size() ) return false;
				m_layers[l][n].m_weight[i] = std::stod(buf);
			}
			ifs >> buf;
		}
	}
	ifs.close();
	return true;
}
