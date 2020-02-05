#include <iostream>
#include <assert.h>
#include <unordered_map>
#include <unordered_set>
#include "utils.h"
#include "HGBoard.h"
#include "HGNNet.h"

using namespace std;

static unordered_set<string> s_set_board;

HGBoard::HGBoard()
{
	m_board.resize(HG_BOARD_SIZE);
	m_black = &m_board[0];
	m_white = &m_board[HG_ARY_SIZE];
	init();
}
//---------------------------------------------------
string Move::text() const {
	string txt = to_string((int)m_src) + "/" + to_string((int)m_src - m_d);
	if (m_hit) txt += '*';
	return txt;
}
//---------------------------------------------------
HGBoard::HGBoard(const HGBoard& x)
{
	m_board = x.m_board;
	m_black = &m_board[0];
	m_white = &m_board[HG_ARY_SIZE];
	m_nBlack = x.m_nBlack;
	m_nWhite = x.m_nWhite;
}
HGBoard::HGBoard(cchar* black, cchar* white)
{
	m_board.resize(HG_BOARD_SIZE);
	m_black = &m_board[0];
	m_white = &m_board[HG_ARY_SIZE];
	m_nBlack = m_nWhite = 0;
	for (int i = HG_GOAL_IX; i != HG_ARY_SIZE; ++i) {
		m_nBlack += (m_black[i] = black[i]);
		m_nWhite += (m_white[i] = white[i]);
	}
}
bool HGBoard::operator==(const HGBoard& x) const {
	return m_board == x.m_board;
}
void HGBoard::init()
{
	clear();
	m_black[3] = m_black[7] = 3;
	m_black[12] = 2;
	m_white[3] = m_white[7] = 3;
	m_white[12] = 2;
	m_nBlack = m_nWhite = 8;
}
void HGBoard::clear()
{
	for (auto& x : m_board) x = 0;
	m_nBlack = m_nWhite = 0;
}
void HGBoard::swapBW()
{
	for (int i = HG_GOAL_IX; i != HG_ARY_SIZE; ++i)
		std::swap(m_black[i], m_white[i]);
	std::swap(m_nBlack, m_nWhite);
}
void HGBoard::set(const std::string& ktext)
{
	m_board = ktext;
	for(auto& x: m_board) x -= '0';
	updateNumBW();
}
void	HGBoard::updateNumBW()				//	白黒石数再計算
{
	m_nBlack = m_nWhite = 0;
	for (int i = HG_START_IX; i >= HG_GOAL_IX; --i) {
		m_nBlack += m_black[i];
		m_nWhite += m_white[i];
	}
}
void HGBoard::b_setAt(int ix, int n)
{
	m_nBlack -= m_black[ix];
	m_nBlack += (m_black[ix] = n);
}
void HGBoard::w_setAt(int ix, int n)
{
	m_nWhite -= m_white[ix];
	m_nWhite += (m_white[ix] = n);
}
static int pips(cchar* ptr) {
	int pips = 0;
	for (int ix = HG_GOAL_IX; ix <= HG_START_IX; ++ix)
		pips += ptr[ix] * ix;
	return pips;
}
int HGBoard::b_pips() const {
	return pips(m_black);
}
int HGBoard::w_pips() const {
	return pips(m_white);
}
int HGBoard::result() const {
	if (m_black[HG_GOAL_IX] == m_nBlack)
		return 1;
	else if (m_white[HG_GOAL_IX] == m_nWhite)
		return -1;
	else
		return 0;
}
int HGBoard::resultSGB() const		//	シングル・ギャモン・バックギャモン勝負を判定、{-3, -2, -1, 0, +1, +2, +3} を返す
{
	if (m_black[HG_GOAL_IX] == m_nBlack) {
		if( m_white[HG_GOAL_IX] == 0 ) {
			if( m_white[HG_START_IX] != 0 || m_white[HG_START_IX-1] != 0 ||
				m_white[HG_START_IX-2] != 0 || m_white[HG_START_IX-3] != 0 )
			{
				return 3;
			}
			if( m_white[HG_START_IX-4] != 0 || m_white[HG_START_IX-5] != 0 || m_white[HG_START_IX-6] != 0 ||
				m_white[HG_START_IX-7] != 0 || m_white[HG_START_IX-8] != 0 || m_white[HG_START_IX-9] != 0 )
			{
				return 2;
			}
		}
		return 1;
	} else if (m_white[HG_GOAL_IX] == m_nWhite) {
		if( m_black[HG_GOAL_IX] == 0 ) {
			if( m_black[HG_START_IX] != 0 || m_black[HG_START_IX-1] != 0 ||
				m_black[HG_START_IX-2] != 0 || m_black[HG_START_IX-3] != 0 )
			{
				return -3;
			}
			if( m_black[HG_START_IX-4] != 0 || m_black[HG_START_IX-5] != 0 || m_black[HG_START_IX-6] != 0 ||
				m_black[HG_START_IX-7] != 0 || m_black[HG_START_IX-8] != 0 || m_black[HG_START_IX-9] != 0 )
			{
				return -2;
			}
		}
		return -1;
	} else
		return 0;
}
std::string HGBoard::ktext() const
{
	string txt = m_board;
	for(auto& x: txt) x += '0';
	return txt;
}
string HGBoard::text() const
{
	//  undone: ヒットされた石表示
	string txt;
	int bp = b_pips(), wp = w_pips();
	txt += "pips black: " + to_string(bp) + "(" + to_string(bp - wp) +
		") white: " + to_string(wp) + "(" + to_string(wp - bp) + ")\n";
	txt += " ";
	if (m_black[HG_START_IX] != 0) {
		txt += "● ";
		txt += to_string((int)m_black[HG_START_IX]);
	}
	else
		txt += "	";
	txt += string(30, ' ');
	//txt += "x ◆";
	if (m_white[HG_START_IX] != 0) {
		txt += to_string((int)m_white[HG_START_IX]);
		txt += " ○";
	}
	else
		txt += "	";
	txt += "\n";
	txt += "+--+--------+--------+--------+--------+--+\n";
	for (int y = 8; y != 0; --y) {
		txt += "|";
		for (int x = HG_START_IX; x >= HG_GOAL_IX; --x) {
			if (x != HG_START_IX && m_black[x] >= y)
				txt += "●";
			else if (hg_revIX(x) != HG_START_IX && m_white[hg_revIX(x)] >= y)
				txt += "○";
			else
				txt += "・";
			if (x == HG_START_IX || x % 3 == 1 || x == HG_GOAL_IX)
				txt += "|";
			else
				txt += " ";
		}
		txt += "\n";
	}
	txt += "+--+--------+--------+--------+--------+--+\n";
	txt += " Ｓ 12 11 10 ９ ８ ７ ６ ５ ４ ３ ２ １ Ｇ\n";
	return txt;
}
void HGBoard::b_genMoves(Moves& mvs, int d) const
{
	//cout << "b_genMoves(d = " << d << ")\n";
	int ix = HG_START_IX;
	while (m_black[ix] == 0) --ix;	 //  黒石は必ず存在すると仮定、なので範囲チェックは行わない
	if (ix == 0) {
		//exit(0);
		//assert(0);
		return;
	}
	//cout << "ix = " << ix << "\n";
	if (ix == HG_START_IX) {
		//cout << "bar\n";
		int wn = m_white[hg_revIX(ix - d)];
		//cout << "wn = " << wn << "\n";
		if (wn < 2) {
			mvs.push_back(Move(ix, d, wn == 1));
		}
	}
	else if (ix <= HG_INNER) {
		//cout << "inner\n";
		bool tail = true;
		for (; ix != 0; --ix, tail = false) {
			if (m_black[ix] != 0) {
				int rx;
				if( ix == d ) {
					mvs.push_back(Move(ix, d));
				} else if (ix > d && m_white[rx = hg_revIX(ix-d)] < 2 ) {
					mvs.push_back(Move(ix, d, m_white[rx] == 1));
				} else if (tail)
					mvs.push_back(Move(ix, ix));
			}
		}
	}
	else {
		while (ix != 0) {
			if (m_black[ix] != 0 && ix - d > 0) {
				int wn = m_white[hg_revIX(ix - d)];
				if (wn < 2)
					mvs.push_back(Move(ix, d, wn == 1));
			}
			--ix;
		}
	}
}
void HGBoard::b_genMovesListSeq(MovesList& lst, int d1, int d2) const
{
	Moves mvs;
	b_genMoves(mvs, d1);
	for (const auto& mv : mvs) {
		HGBoard b2(*this);
		b2.b_move(mv);
		Moves mvs2;
		b2.b_genMoves(mvs2, d2);
		if (mvs2.empty()) {
			lst.push_back(Moves());
			lst.back().push_back(mv);
		}
		else {
			for (auto mv2 : mvs2) {
				HGBoard b3(b2);
				b3.b_move(mv2);
				if (s_set_board.find(b3.m_board) == s_set_board.end()) {
					s_set_board.insert(b3.m_board);
					lst.push_back(Moves());
					lst.back().push_back(mv);
					lst.back().push_back(mv2);
				}
			}
		}
	}
}
void HGBoard::b_genMovesList(MovesList& lst, int d1, int d2) const
{
	s_set_board.clear();
	lst.clear();
	if (d1 != d2) {	//  ゾロ目ではない場合
		b_genMovesListSeq(lst, d1, d2);
		b_genMovesListSeq(lst, d2, d1);
	}
	else {			//  ゾロ目の場合
		Moves mvs1;
		b_genMoves(mvs1, d1);
		for (const auto& mv1 : mvs1) {
			HGBoard b2(*this);
			b2.b_move(mv1);
			Moves mvs2;
			b2.b_genMoves(mvs2, d1);
			if (mvs2.empty()) {
				if (s_set_board.find(b2.m_board) == s_set_board.end()) {
					s_set_board.insert(b2.m_board);
					lst.push_back(Moves());
					lst.back().push_back(mv1);
				}
			}
			else {
				for (auto mv2 : mvs2) {
					HGBoard b3(b2);
					b3.b_move(mv2);
					Moves mvs3;
					b3.b_genMoves(mvs3, d1);
					if (mvs3.empty()) {
						if (s_set_board.find(b3.m_board) == s_set_board.end()) {
							s_set_board.insert(b3.m_board);
							lst.push_back(Moves());
							lst.back().push_back(mv1);
							lst.back().push_back(mv2);
						}
					}
					else {
						for (auto mv3 : mvs3) {
							HGBoard b4(b3);
							b4.b_move(mv3);
							Moves mvs4;
							b4.b_genMoves(mvs4, d1);
							if (mvs4.empty()) {
								if (s_set_board.find(b4.m_board) == s_set_board.end()) {
									s_set_board.insert(b4.m_board);
									lst.push_back(Moves());
									lst.back().push_back(mv1);
									lst.back().push_back(mv2);
									lst.back().push_back(mv3);
								}
							}
							else {
								for (auto mv4 : mvs4) {
									HGBoard b5(b4);
									b5.b_move(mv4);
									if (s_set_board.find(b5.m_board) == s_set_board.end()) {
										s_set_board.insert(b5.m_board);
										lst.push_back(Moves());
										lst.back().push_back(mv1);
										lst.back().push_back(mv2);
										lst.back().push_back(mv3);
										lst.back().push_back(mv4);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
void HGBoard::w_genMovesList(MovesList& lst, int d1, int d2) const
{
	HGBoard b2(m_white, m_black);
	b2.b_genMovesList(lst, d1, d2);
}
void HGBoard::b_move(const Move& mv) {
	int dst = mv.m_src - mv.m_d;
	if (dst < 0) dst = HG_GOAL_IX;
	m_black[(int)mv.m_src] -= 1;
	int rx = hg_revIX(dst);
	if (mv.m_hit) {	//  hit
		m_white[rx] = 0;
		m_white[HG_START_IX] += 1;
		//mv.m_hit = true;
	}
	m_black[dst] += 1;
}
void HGBoard::w_move(const Move& mv) {
	int dst = mv.m_src - mv.m_d;
	if (dst < 0) dst = HG_GOAL_IX;
	m_white[(int)mv.m_src] -= 1;
	int rx = hg_revIX(dst);
	if (mv.m_hit) {	//  hit
		m_black[rx] = 0;
		m_black[HG_START_IX] += 1;
		//mv.m_hit = true;
	}
	m_white[dst] += 1;
}
void HGBoard::b_move(const vector<Move>& mvs) {
	for (const auto& mv : mvs) b_move(mv);
}
void HGBoard::w_move(const vector<Move>& mvs) {
	for (const auto& mv : mvs) w_move(mv);
}
void HGBoard::b_unMove(const Move& mv) {
	int dst = mv.m_src - mv.m_d;
	if (dst < 0) dst = HG_GOAL_IX;
	m_black[dst] -= 1;
	int rx = hg_revIX(dst);
	if (mv.m_hit) {	//  hit
		m_white[rx] = 1;
		m_white[HG_START_IX] -= 1;
		//mv.m_hit = true;
	}
	m_black[(int)mv.m_src] += 1;
}
void HGBoard::w_unMove(const Move& mv) {
	int dst = mv.m_src - mv.m_d;
	if (dst < 0) dst = HG_GOAL_IX;
	m_white[dst] -= 1;
	int rx = hg_revIX(dst);
	if (mv.m_hit) {	//  hit
		m_black[rx] = 1;
		m_black[HG_START_IX] -= 1;
		//mv.m_hit = true;
	}
	m_white[(int)mv.m_src] += 1;
}
//	ランダムプレイアウトによる得点期待値計算、黒番固定
//	リターン値がプラスならば黒有利
double HGBoard::b_expScoreRPO(int N_GAME) const
{
	int sum = 0;
	for (int i = 0; i < N_GAME; ++i) {
		HGBoard b2(*this);
		int resultSGB;
		for(bool bt = true;; bt = !bt) {
			MovesList lst;
			int d1 = g_mt() % 3 + 1;
			int d2 = g_mt() % 3 + 1;
			if( bt ) {
				b2.b_genMovesList(lst, d1, d2);
				if( !lst.empty() )
					b2.b_move(lst[g_mt() % lst.size()]);
			} else {
				b2.w_genMovesList(lst, d1, d2);
				if( !lst.empty() )
					b2.w_move(lst[g_mt() % lst.size()]);
			}
			if( (resultSGB = b2.resultSGB()) != 0 ) break;
		}
		sum += resultSGB;
	}
	return (double)sum / N_GAME;
}
//	ランダムプレイアウトによる得点期待値計算
//	リターン値がプラスならば白有利
double HGBoard::w_expScoreRPO(int N_GAME) const
{
	HGBoard b2(m_white, m_black);
	return b2.b_expScoreRPO(N_GAME);
}
void HGBoard::setInput(vector<double>& input) const
{
	input.resize(HG_NN_INSIZE);
	int ix = 0;
	input[ix++] = m_black[HG_GOAL_IX] / 8.0;
	input[ix++] = -m_white[HG_GOAL_IX] / 8.0;
	for (int i = 1; i <= HG_N_POINT; ++i) {
		input[ix++] = m_black[i] == 0 ? 0 : 1;
		input[ix++] = m_black[i] <= 1 ? 0 : 1;
		input[ix++] = m_black[i] <= 2 ? 0 : 1;
		input[ix++] = m_black[i] <= 3 ? 0 : m_black[i] / 8.0;
		input[ix++] = m_white[i] == 0 ? 0 : -1;
		input[ix++] = m_white[i] <= 1 ? 0 : -1;
		input[ix++] = m_white[i] <= 2 ? 0 : -1;
		input[ix++] = m_white[i] <= 3 ? 0 : -m_black[i] / 8.0;
	}
	input[ix++] = m_black[HG_START_IX] / 2.0;
	input[ix++] = -m_white[HG_START_IX] / 2.0;
}
void HGBoard::setInputNmlz(std::vector<double>& input) const				//	平均０、分散１に変換
{
	input.resize(HG_NN_INSIZE);
	double sum = 0, sum2 = 0, d;
	int ix = 0;
	sum += d = input[ix++] = m_black[HG_GOAL_IX] / 8.0;
	sum2 += d * d;
	sum += d = input[ix++] = -m_white[HG_GOAL_IX] / 8.0;
	sum2 += d * d;
	for (int i = 1; i <= HG_N_POINT; ++i) {
		sum += d = input[ix++] = m_black[i] == 0 ? 0 : 1;
		sum2 += d * d;
		sum += d = input[ix++] = m_black[i] <= 1 ? 0 : 1;
		sum2 += d * d;
		sum += d = input[ix++] = m_black[i] <= 2 ? 0 : 1;
		sum2 += d * d;
		sum += d = input[ix++] = m_black[i] <= 3 ? 0 : m_black[i] / 8.0;
		sum2 += d * d;
		sum += d = input[ix++] = m_white[i] == 0 ? 0 : -1;
		sum2 += d * d;
		sum += d = input[ix++] = m_white[i] <= 1 ? 0 : -1;
		sum2 += d * d;
		sum += d = input[ix++] = m_white[i] <= 2 ? 0 : -1;
		sum2 += d * d;
		sum += d = input[ix++] = m_white[i] <= 3 ? 0 : -m_black[i] / 8.0;
		sum2 += d * d;
	}
	sum += d = input[ix++] = m_black[HG_START_IX] / 2.0;
	sum2 += d * d;
	sum += d = input[ix++] = -m_white[HG_START_IX] / 2.0;
	sum2 += d * d;
	double ave = sum / ix;
	double std2 = sum2 / ix - ave * ave;
	for(auto& x: input) x = (x - ave) / sqrt(std2 + 1e-6);
}
//	黒番・１手先読み・HGNNet による得点期待値により最適手取得
void HGBoard::negaMax1(Moves& mvs, class HGNNet& nn, int d1, int d2) const
{
	MovesList lst;
	b_genMovesList(lst, d1, d2);
	if( lst.empty() ) {
		mvs.clear();
		return;
	}
	double mxev = -9999;
	int mxi = 0;
	vector<double> input;
	int i = 0;
	for (; i != lst.size(); ++i) {
		HGBoard b2(*this);
		b2.b_move(lst[i]);
		b2.swapBW();
		b2.setInput(input);
		double ev = -nn.predict(input);
		if( ev > mxev ) {
			mxev = ev;
			mxi = i;
		}
	}
	mvs = lst[mxi];
}
