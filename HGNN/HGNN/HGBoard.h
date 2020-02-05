#pragma once

#include <string>
#include <vector>

//class HGNNet;

#define	 HG_N_POINT		 12
#define	 HG_ARY_SIZE		(HG_N_POINT+2)	 //  +2 for スタート、ゴール
#define	 HG_BOARD_SIZE	  HG_ARY_SIZE*2
#define	 HG_START_IX		(HG_ARY_SIZE-1)
#define	 HG_GOAL_IX		 0
#define	 HG_INNER		   3			   //  インナー [1, 3]
#define	 HG_MAX_DICE		3			   //  ダイスの目：[1, 3]

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
	char	m_src;	  //  移動元
	char	m_d;		//  移動距離
	bool	m_hit;	  //  ヒット？
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
	int	result() const;		//	{-1, 0, +1} を返す
	int	resultSGB() const;		//	ギャモン・バックギャモン勝負を判定、{-3, -2, -1, 0, +1, +2, +3} を返す
	void	b_genMoves(Moves&, int) const;
	void	b_genMovesListSeq(MovesList&, int, int) const;  //  d1 != d2, d1, d2 の順に使用
	void	b_genMovesList(MovesList&, int, int) const;
	void	w_genMovesList(MovesList&, int, int) const;
	double	b_expScoreRPO(int N_GAME = 100) const;			//	ランダムプレイアウトによる得点期待値計算、リターン値がプラスならば黒有利
	double	w_expScoreRPO(int N_GAME = 100) const;			//	ランダムプレイアウトによる得点期待値計算、リターン値がプラスならば白有利
	void	setInput(std::vector<double>&) const;
	void	setInputNmlz(std::vector<double>&) const;				//	平均０、分散１に変換
	double	b_expctScore(class HGNNet&) const;					//	黒番 スコア期待値
	double	w_expctScore(class HGNNet&) const;					//	白番 スコア期待値
	void	negaMax1(Moves&, class HGNNet&, int, int) const;					//	黒番・１手先読み・HGNNet による得点期待値により最適手取得
public:
	void	init();
	void	clear();
	void	swapBW();
	void	updateNumBW();				//	白黒石数再計算
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
	char* m_black;	   //  黒石、m_board[0] を指す
	char* m_white;	   //  白石、m_board[HG_ARY_SIZE] を指す
	int	 m_nBlack;
	int	 m_nWhite;
	std::string  m_board;		//  白黒石データ、size = HG_ARY_SIZE*2
};