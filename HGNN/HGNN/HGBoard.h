#pragma once

#include <string>
#include <vector>

//class HGNNet;

#define	 HG_N_POINT		 12
#define	 HG_ARY_SIZE		(HG_N_POINT+2)	 //  +2 for �X�^�[�g�A�S�[��
#define	 HG_BOARD_SIZE	  HG_ARY_SIZE*2
#define	 HG_START_IX		(HG_ARY_SIZE-1)
#define	 HG_GOAL_IX		 0
#define	 HG_INNER		   3			   //  �C���i�[ [1, 3]
#define	 HG_MAX_DICE		3			   //  �_�C�X�̖ځF[1, 3]

#define		HG_NN_INSIZE		(HG_N_POINT*2*4+2+2)
#define		HG_NN_HIDSIZE		40
//#define		HG_NN_HIDSIZE		60
//#define		HG_NN_HIDSIZE		80

static inline int hg_revIX(int x) { return HG_START_IX - x; }

typedef const char cchar;

struct Move {
public:
	Move(char src = 0, char d = 0, bool hit = false)
		: m_src(src)
		, m_d(d)
		, m_hit(hit)
	{
	}
public:
	std::string text() const;
public:
	char	m_src;	  //  �ړ���
	char	m_d;		//  �ړ�����
	bool	m_hit;	  //  �q�b�g�H
};

typedef std::vector<Move> Moves;
typedef std::vector<Moves> MovesList;

class HGBoard {
public:
	HGBoard();
	HGBoard(const HGBoard&);
	HGBoard(cchar* black, cchar* white);
public:
	bool	operator==(const HGBoard&) const;
	bool	operator!=(const HGBoard& x) const { return !operator==(x); }
	std::string board() const { return m_board; }
	//std::string key() const { return m_board; }
	std::string ktext() const;
	std::string text() const;
	int	b_pips() const;
	int	w_pips() const;
	int	result() const;		//	{-1, 0, +1} ��Ԃ�
	int	resultSGB() const;		//	�M�������E�o�b�N�M�����������𔻒�A{-3, -2, -1, 0, +1, +2, +3} ��Ԃ�
	void	b_genMoves(Moves&, int) const;
	void	b_genMovesListSeq(MovesList&, int, int) const;  //  d1 != d2, d1, d2 �̏��Ɏg�p
	void	b_genMovesList(MovesList&, int, int) const;
	void	w_genMovesList(MovesList&, int, int) const;
	double	b_expScoreRPO(int N_GAME = 100) const;			//	�����_���v���C�A�E�g�ɂ�链�_���Ғl�v�Z�A���^�[���l���v���X�Ȃ�΍��L��
	double	w_expScoreRPO(int N_GAME = 100) const;			//	�����_���v���C�A�E�g�ɂ�链�_���Ғl�v�Z�A���^�[���l���v���X�Ȃ�Δ��L��
	void	setInput(std::vector<double>&) const;
	void	setInputNmlz(std::vector<double>&) const;				//	���ςO�A���U�P�ɕϊ�
	double	b_expctScore(class HGNNet&) const;					//	���� �X�R�A���Ғl
	double	w_expctScore(class HGNNet&) const;					//	���� �X�R�A���Ғl
	void	negaMax1(Moves&, class HGNNet&, int, int) const;					//	���ԁE�P���ǂ݁EHGNNet �ɂ�链�_���Ғl�ɂ��œK��擾
public:
	void	init();
	void	clear();
	void	swapBW();
	void	updateNumBW();				//	�����ΐ��Čv�Z
	void	set(const std::string&);
	void	b_setAt(int ix, int n);
	void	w_setAt(int ix, int n);
	void	b_move(const Move&);
	void	w_move(const Move&);
	void	b_move(const std::vector<Move>&);
	void	w_move(const std::vector<Move>&);
	void	b_unMove(const Move&);
	void	w_unMove(const Move&);
//private:
public:
	char* m_black;	   //  ���΁Am_board[0] ���w��
	char* m_white;	   //  ���΁Am_board[HG_ARY_SIZE] ���w��
	int	 m_nBlack;
	int	 m_nWhite;
	std::string  m_board;		//  �����΃f�[�^�Asize = HG_ARY_SIZE*2
};