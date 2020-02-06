//----------------------------------------------------------------------
//
//		�n�[�t�M�����������j���[�����l�b�g���[�N
//		HGNNet �N���X�����t�@�C��
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
static std::uniform_real_distribution<> g_rand11(-1, +1);        // [-1, +1] �͈͂̈�l����
#endif

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double tanhDiff(double x) {
	auto ch = cosh(x);
	return 1.0 / (ch * ch);
}
const double LR_DIFF = 0.01;		//	Leaky-ReLU(x) x<0 �̏ꍇ�̌X�� 
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
//	��P�����F���́E�B��w���C���[�̃m�[�h�����X�g�A�o�̓��C���[�͎w�肵�Ȃ�
//	��Q�����F�������֐����
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
	m_nInput = lst[0];		//	���͑w�m�[�h��
	m_layers.resize(lst.size());
	//	�B��w������
	for (int l = 1; l != lst.size(); ++l) {
		auto& layer = m_layers[l - 1];
		layer.resize(lst[l]);
		for (int n = 0; n != lst[l]; ++n) {
			layer[n].m_weight.resize(lst[l - 1] + 1);		//	+1 for �o�C�A�X
			for (auto& w : layer[n].m_weight) w = g_rand11(g_mt);		//	�W�� [-1, +1] ����������
			layer[n].m_wtDiff.resize(lst[l - 1] + 1);		//	+1 for �o�C�A�X
			for (auto& w : layer[n].m_wtDiff) w = 0;		//	
		}
	}
	//	�o�͑w������
	int l = lst.size() - 1;
	auto& layer = m_layers[l];		//	
	layer.resize(1);
	if (l != 0) {	//	�B��w������ꍇ
		layer[0].m_weight.resize(lst[l] + 1);		//	+1 for �o�C�A�X
		layer[0].m_wtDiff.resize(lst[l] + 1);		//	+1 for �o�C�A�X
	} else {			//	�B��w�������ꍇ
		layer[0].m_weight.resize(lst[0] + 1);		//	+1 for �o�C�A�X
		layer[0].m_wtDiff.resize(lst[0] + 1);		//	+1 for �o�C�A�X
	}
	for (auto& w : layer[0].m_weight) w = g_rand11(g_mt);		//	�W�� [-1, +1] ����������
	for (auto& w : layer[0].m_wtDiff) w = 0;
}
//	���� �� ��A�\��
data_t HGNNet::predict(const std::vector<data_t>& input)			//	�\��
{
	if (input.size() != m_nInput) {
		//cout << "input.size() = " << input.size() << ", m_nInput = " << m_nInput << "\n";
		assert(0);
		return 0;
	}
	for (int l = 0; l != m_layers.size(); ++l) {		//	for �e���C���[
		const bool outLayer = l == m_layers.size() - 1;
		vector<HGNNNode>& layer = m_layers[l];
		double sum = 0, sum2 = 0;		//	for �o�b�`�E�m�[�}���C�[�[�V����
		for (int n = 0; n != layer.size(); ++n) {		//	���C���[�̑S�m�[�h������
			HGNNNode& node = layer[n];
			node.m_sum = node.m_weight.back();		//	�o�C�A�X
			if (l == 0) {		//	�O�i�����͑w�̏ꍇ
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
				node.m_diff = (1 - node.m_output) * node.m_output;		//	�����l
				assert( !isnan(node.m_output) );
				break;
			case TANH:
				node.m_output = tanh(node.m_sum);
				node.m_diff = tanhDiff(node.m_sum);		//	�����l
				assert( !isnan(node.m_output) );
				break;
			case RELU:
				//node.m_output = std::max(0.0, node.m_sum);
				node.m_output = node.m_sum >= 0 ? node.m_sum : 0;
				node.m_diff = node.m_sum >= 0 ? 1 : 0;		//	�����l
				assert( !isnan(node.m_output) );
				break;
			case LEAKY_RELU:
				node.m_output = node.m_sum >= 0 ? node.m_sum : node.m_sum * LR_DIFF;
				node.m_diff = node.m_sum >= 0 ? 1 : LR_DIFF;		//	�����l
				assert( !isnan(node.m_output) );
				break;
			default:
				node.m_output = node.m_sum;		//	�P���֐�
				node.m_diff = 1.0;			//	f(x) = x �̔����� 1.0
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
			for (int n = 0; n != layer.size(); ++n) {		//	���C���[�̑S�m�[�h������
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
//	�w�K�A��P�����F���͒l�A��Q�����F���t�l
void HGNNet::learn(const std::vector<data_t>& input, data_t t, double alpha)
{
	const double ETA = 0.9;
	if( alpha > 0 ) m_alpha = alpha;
#if	1
	calcError(input, t);
	for (int l = m_layers.size(); --l >= 0;) {		//	�S���C���[�ɂ���
		HGNNLayer& layer = m_layers[l];
		for (int n = 0; n != layer.size(); ++n) {
			HGNNNode& node = layer[n];
			if (l > 0) {		//	�O�i���B��w�̏ꍇ
				auto& prevLayer = m_layers[l - 1];
				for (int k = 0; k != prevLayer.size(); ++k) {		//	�O�i�̑S�m�[�h�ɂ���
					auto& prevNode = prevLayer[k];
					if( m_optSGD ) {
						auto t = node.m_wtDiff[k] * ETA;
						node.m_weight[k] -= t + (node.m_wtDiff[k] = m_alpha * node.m_err * prevLayer[k].m_output);	//	�d�ݏC��
					} else
						node.m_weight[k] -= m_alpha * node.m_err * prevLayer[k].m_output;	//	�d�ݏC��
					assert( !isnan(node.m_weight[k]) );
					assert( abs(node.m_weight[k]) < 1e12 );
				}
			} else {			//	�O�i�����͑w�̏ꍇ
				for (int k = 0; k != node.m_weight.size() - 1; ++k) {		//	-�d�݌W���z��̍Ō�̓o�C�A�X�p�̂��߂� -1
					if( m_optSGD ) {
						auto t = node.m_wtDiff[k] * ETA;
						node.m_weight[k] -= t + (node.m_wtDiff[k] = m_alpha * node.m_err * input[k]);	//	�d�ݏC��
					} else
						node.m_weight[k] -= m_alpha * node.m_err * input[k];
					assert( !isnan(node.m_weight[k]) );
					assert( abs(node.m_weight[k]) < 1e12 );
				}
			}
			if( m_optSGD ) {
				auto t = node.m_wtDiff.back() * ETA;
				node.m_weight.back() -= t + (node.m_wtDiff.back() = m_alpha * node.m_err);	//	�d�ݏC��
			} else
				node.m_weight.back() -= m_alpha * node.m_err;	//	for �o�C�A�X
			assert( !isnan(node.m_weight.back()) );
			assert( abs(node.m_weight.back()) < 1e12 );
		}
	}
#else
	for (auto& layer : m_layers) {
		for (auto& node : layer)
			node.m_err = 0;			//	���ׂĂ̌덷����������N���A
	}
	//
	data_t y = predict(input);
	//m_layers.back().front().m_err = y - t;			//	�o�̓m�[�h�덷
	for (int l = m_layers.size(); --l >= 0;) {		//	�S���C���[�ɂ���
		HGNNLayer& layer = m_layers[l];
		for (int n = 0; n != layer.size(); ++n) {
			HGNNNode& node = layer[n];
			if (l == m_layers.size() - 1) {	//	�o�͑w�̏ꍇ
				if( m_outputType == OT_TANH )
					node.m_err = y - log((1+t)/(1-t)) / 2;			//	��L/��y�AL = (y-t)^2/2
				else
					node.m_err = y - t;			//	��L/��y�AL = (y-t)^2/2
			}
			if (l > 0) {		//	�O�i���B��w�̏ꍇ
				auto& prevLayer = m_layers[l - 1];
				//for (int i = 0; i != node.m_weight.size() - 1; ++i) {		//	-�d�݌W���z��̍Ō�̓o�C�A�X�p�̂��߂� -1
				for (int k = 0; k != prevLayer.size(); ++k) {		//	�O�i�̑S�m�[�h�ɂ���
					auto& prevNode = prevLayer[k];
					//cout	<< "l, n, k = " << l << ", " << n << ", " << k << ":\n";
					//cout	<< "  prevNode.m_err = " << prevNode.m_err
					//		<< ", prevNode.m_diff = " << prevNode.m_diff << "\n";
					//cout	<< "  node.m_err = " << node.m_err
					//		<< ", node.m_weight = " << node.m_weight[k]<< "\n";
					prevNode.m_err += prevNode.m_diff * node.m_err * node.m_weight[k];		//	�덷�t�`��
					//cout	<< "  --> " << prevNode.m_err << "\n";
					node.m_weight[k] -= m_alpha * node.m_err * prevLayer[k].m_output;	//	�d�ݏC��
				}
				//node.m_weight[i] -= m_alpha * node.m_err * m_layers[l-1][i].m_output;	//	�d�ݏC��
			//}
			}
			else {			//	�O�i�����͑w�̏ꍇ
				for (int i = 0; i != node.m_weight.size() - 1; ++i) {		//	-�d�݌W���z��̍Ō�̓o�C�A�X�p�̂��߂� -1
					node.m_weight[i] -= m_alpha * node.m_err * input[i];
				}
			}
			node.m_weight.back() -= m_alpha * node.m_err;	//	for �o�C�A�X
		}
	}
#endif
}
//	�덷�t�`���v�Z�̂� for Test
void HGNNet::calcError(const std::vector<data_t>& input, data_t t)
{
	for (auto& layer : m_layers) {
		for (auto& node : layer)
			node.m_err = 0;			//	���ׂĂ̌덷����������N���A
	}
	//
	data_t y = predict(input);
	//m_layers.back().front().m_err = y - t;			//	�o�̓m�[�h�덷
	for (int l = m_layers.size(); --l >= 0;) {		//	�S���C���[�ɂ���
		HGNNLayer& layer = m_layers[l];
		for (int n = 0; n != layer.size(); ++n) {
			HGNNNode& node = layer[n];
			assert( abs(node.m_err) < 1e12 );
			if (l == m_layers.size() - 1) {	//	�o�͑w�̏ꍇ
				if( m_outputType == OT_TANH )
					node.m_err = y - log((1+t)/(1-t)) / 2;			//	��L/��y�AL = (y-t)^2/2
				else
					node.m_err = y - t;			//	��L/��y�AL = (y-t)^2/2
				assert( abs(node.m_err) < 1e12 );
			}
			if (l > 0) {		//	�O�i���B��w�̏ꍇ
				auto& prevLayer = m_layers[l - 1];
				//for (int i = 0; i != node.m_weight.size() - 1; ++i) {		//	-�d�݌W���z��̍Ō�̓o�C�A�X�p�̂��߂� -1
				for (int k = 0; k != prevLayer.size(); ++k) {		//	�O�i�̑S�m�[�h�ɂ���
					auto& prevNode = prevLayer[k];
					//cout	<< "l, n, k = " << l << ", " << n << ", " << k << ":\n";
					//cout	<< "  prevNode.m_err = " << prevNode.m_err
					//		<< ", prevNode.m_diff = " << prevNode.m_diff << "\n";
					//cout	<< "  node.m_err = " << node.m_err
					//		<< ", node.m_weight = " << node.m_weight[k]<< "\n";
					prevNode.m_err += prevNode.m_diff * node.m_err * node.m_weight[k];		//	�덷�t�`��
					//cout	<< "  --> " << prevNode.m_err << "\n";
					//node.m_weight[k] -= m_alpha * node.m_err * prevLayer[k].m_output;	//	�d�ݏC��
					assert( abs(prevNode.m_err) < 1e12 );
				}
				//node.m_weight[i] -= m_alpha * node.m_err * m_layers[l-1][i].m_output;	//	�d�ݏC��
			//}
			} else {			//	�O�i�����͑w�̏ꍇ
				//for (int i = 0; i != node.m_weight.size() - 1; ++i) {		//	-�d�݌W���z��̍Ō�̓o�C�A�X�p�̂��߂� -1
				//	node.m_weight[i] -= m_alpha * node.m_err * input[i];
				//}
			}
			//node.m_weight.back() -= m_alpha * node.m_err;	//	for �o�C�A�X
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
std::string HGNNet::dumpWeight(bool highPrec) const		//	�d�݌W���̂�
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
void HGNNet::makeWeightSeq()			//	�W���� 0.1, 0.2, ... �ɐݒ�Afor Test
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
//	for Test�A�o�b�N�v���p�Q�[�V�����̂��߂̌덷�E�����l��\��
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
bool HGNNet::save(cchar* fname) const		//	�w��t�@�C���Ƀl�b�g���[�N�̏�Ԃ�ۑ�
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
	if( lst.back() != 1 ) return false;		//	�o�͑w�͉�A�̂݃T�|�[�g
	lst.pop_back();		//	remove �o�͑w
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
