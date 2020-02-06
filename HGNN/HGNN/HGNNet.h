//----------------------------------------------------------------------
//
//		�n�[�t�M�����������j���[�����l�b�g���[�N
//		HGNNet �N���X�錾�t�@�C��
//		Copyright (C) 2020 by N.Tsuda
//		License: MIT
//
//----------------------------------------------------------------------

#pragma once

#include <vector>
#include <string>

/*

	�n�[�t�M�������p�j���[�����l�b�g�N���X�錾

	�������֐��Fsigmoid(), tanh(), ReLU() �̂݃T�|�[�g�A�B��w�������֐��͂��ׂē���
	�o�͉͂�A�̂�

*/

//typedef double HGNND_t;
typedef double data_t;					//	NN�f�[�^�^�C�v
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
	data_t	m_output;		//	�o�͒l�AactFunc(m_sum)
	data_t	m_sum;			//	�����͒l*�d�݌W��
	data_t	m_diff;			//	�������֐������l�AactFunc'(m_sum)
	data_t	m_err;			//	�덷 for �o�b�N�v���p�Q�[�V���� = ��m_diff*��i�덷*��i�d�݌W��
	std::vector<data_t>	m_weight;		//	�O�i�Ƃ̏d�݌W���A�Ō�̗v�f�̓o�C�A�X�p
	std::vector<data_t>	m_wtDiff;		//	�d�݌W������
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
	std::string	dumpWeight(bool = true) const;		//	�d�݌W���̂�
	std::string	dumpPredict(const std::vector<data_t>& input);		//	for Test�A
	std::string	dumpBP() const;		//	for Test�A�o�b�N�v���p�Q�[�V�����̂��߂̌덷�E�����l��\��
	bool	save(cchar*) const;		//	�w��t�@�C���Ƀl�b�g���[�N�̏�Ԃ�ۑ�
public:
	//	������
	//		��P�����F���́E�B��w���C���[�̃��j�b�g�����X�g�A�o�̓��C���[�͎w�肵�Ȃ�
	//		��Q�����F�������֐����
	void	init(const std::vector<int>&, ActFunc, bool batchNrmlz, double alpha = 0.01);
	void	init(const std::vector<int>&, ActFunc, double alpha = 0.01);
	void	init(const std::vector<int>&, ActFunc, OutputType, double alpha = 0.01);
	//	�\��
	data_t	predict(const std::vector<data_t>& input);
	//	�w�K�A��P�����F���͒l�A��Q�����F���t�l
	void	learn(const std::vector<data_t>& input, data_t t, double alpha = -1);
	void	train(const std::vector<data_t>& input, data_t t, double alpha = -1) { learn(input, t, alpha); }		//	�G�C���A�X
	//	�w�K�A��P�����F���͒l�A��Q�����F���t�l
	void	calcError(const std::vector<data_t>& input, data_t t);		//	�덷�t�`���v�Z�̂� for Test
	bool	load(cchar*);
	//bool	save(cchar*);
	void	makeWeightSeq();			//	�W���� 0.1, 0.2, ... �ɐݒ�Afor Test
public:
	bool			m_batchNrmlz;		//	�o�b�`�E�m�[�}���C�[�[�V����
	bool			m_optSGD;			//	���z�~���@�œK��
	uint8			m_outputType;		//	��A/tanh
	int			m_nInput;				//	���̓��C���[�̃m�[�h��
	double		m_alpha;				//	�w�K�W���A�͈́F[0, 1]
	ActFunc		m_actFunc;				//	�B��w �������֐����
	std::vector<HGNNLayer>	m_layers;		//	���́E�B��E�o�̓��C���[
};
