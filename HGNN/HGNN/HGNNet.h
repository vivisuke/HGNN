//----------------------------------------------------------------------
//
//		ハーフギャモン向けニューラルネットワーク
//		HGNNet クラス宣言ファイル
//		Copyright (C) 2020 by N.Tsuda
//		License: MIT
//
//----------------------------------------------------------------------

#pragma once

#include <vector>
#include <string>

/*

	ハーフギャモン用ニューラルネットクラス宣言

	活性化関数：sigmoid(), tanh(), ReLU() のみサポート、隠れ層活性化関数はすべて同じ
	出力は回帰のみ

*/

//typedef double HGNND_t;
typedef double data_t;					//	NNデータタイプ
typedef const char cchar;
typedef unsigned char uint8;

enum ActFunc {
	NONE = 0,
	SIGMOID,
	TANH,
	RELU,
	LEAKY_RELU,
};
enum OutputType {
	OT_REGRESSION = 0,
	OT_TANH,						//	[-1, +1]
};

struct HGNNNode {
	data_t	m_output;		//	出力値、actFunc(m_sum)
	data_t	m_sum;			//	∑入力値*重み係数
	data_t	m_diff;			//	活性化関数微分値、actFunc'(m_sum)
	data_t	m_err;			//	誤差 for バックプロパゲーション = ∑m_diff*後段誤差*後段重み係数
	std::vector<data_t>	m_weight;		//	前段との重み係数、最後の要素はバイアス用
	std::vector<data_t>	m_wtDiff;		//	重み係数差分
};

typedef std::vector<HGNNNode> HGNNLayer;
//struct HGNNLayer {
//	std::vector<HGNNNode>	m_nodes;
//};

class HGNNet {
public:
	HGNNet();
public:
	bool operator==(const HGNNet&) const;
	std::string	dump() const;
	std::string	dumpWeight(bool = true) const;		//	重み係数のみ
	std::string	dumpPredict(const std::vector<data_t>& input);		//	for Test、
	std::string	dumpBP() const;		//	for Test、バックプロパゲーションのための誤差・微分値を表示
	bool	save(cchar*) const;		//	指定ファイルにネットワークの状態を保存
public:
	//	初期化
	//		第１引数：入力・隠れ層レイヤーのユニット数リスト、出力レイヤーは指定しない
	//		第２引数：活性化関数種別
	void	init(const std::vector<int>&, ActFunc, bool batchNrmlz, double alpha = 0.01);
	void	init(const std::vector<int>&, ActFunc, double alpha = 0.01);
	void	init(const std::vector<int>&, ActFunc, OutputType, double alpha = 0.01);
	//	予測
	data_t	predict(const std::vector<data_t>& input);
	//	学習、第１引数：入力値、第２引数：教師値
	void	learn(const std::vector<data_t>& input, data_t t, double alpha = -1);
	void	train(const std::vector<data_t>& input, data_t t, double alpha = -1) { learn(input, t, alpha); }		//	エイリアス
	//	学習、第１引数：入力値、第２引数：教師値
	void	calcError(const std::vector<data_t>& input, data_t t);		//	誤差逆伝搬計算のみ for Test
	bool	load(cchar*);
	//bool	save(cchar*);
	void	makeWeightSeq();			//	係数を 0.1, 0.2, ... に設定、for Test
public:
	bool			m_batchNrmlz;		//	バッチ・ノーマライゼーション
	bool			m_optSGD;			//	勾配降下法最適化
	uint8			m_outputType;		//	回帰/tanh
	int			m_nInput;				//	入力レイヤーのノード数
	double		m_alpha;				//	学習係数、範囲：[0, 1]
	ActFunc		m_actFunc;				//	隠れ層 活性化関数種別
	std::vector<HGNNLayer>	m_layers;		//	入力・隠れ・出力レイヤー
};
