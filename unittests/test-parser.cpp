#include "basicOp.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

using namespace std;
using namespace mlir::linnea::expr;

namespace parser {
/// POD to contain parsed operand information.
struct ParsedOperand {
  string name;
  string type;
  SmallVector<int64_t, 2> dims;
  vector<Expr::ExprProperty> properties;
  void dump();
  static Expr::ExprProperty convertProperty(string str);
};

Expr::ExprProperty ParsedOperand::convertProperty(string str) {
  if (str.compare("SYMMETRIC") == 0)
    return Expr::ExprProperty::SYMMETRIC;
  if (str.compare("SQUARE") == 0)
    return Expr::ExprProperty::SQUARE;
  llvm_unreachable("Cannot convert property");
}

void ParsedOperand::dump() {
  cout << "printing parsed operand->\n";
  cout << "name: " << name << "\n";
  cout << "type: " << type << "\n";
  cout << "#dims: " << dims.size() << "\n";
  for (auto dim : dims)
    cout << dim << " ";
  cout << "\n";
}
} // end namespace parser.

namespace lexer {

enum class TokenValue {
  TK_DEF,   // def
  TK_WHERE, // where
  TK_IS,    // is
  NAME,     // a name identifier
  LP,       // (
  RP,       // )
  NUM,      // a number
  COMMA,    // ,
  CLP,      // {
  CRP,      // }
  EOS,      // end of string
  EQ,       // =
  MUL,      // *
  ADD       // +
};

/// Terminator POD for Trie data structure.
struct Terminator {
  Terminator() : isEnd(false), tokVal(TokenValue::EOS){};
  Terminator(TokenValue tokVal) : isEnd(true), tokVal(tokVal){};
  bool isEnd = false;
  TokenValue tokVal = TokenValue::EOS;
};

/// Trie data structure to match in O(maxlength) keywords
/// exposed by the language.
class TrieNode {
public:
  TrieNode() = default;
  TrieNode(TokenValue tokVal) : terminator(Terminator(tokVal)){};

  // insert 'str' in the tree with size 'size'. The terminator
  // node in the tree for 'str' will have the tokenValue 'tokval'.
  void insertImpl(string str, int idx, int size, TokenValue tokVal);
  void insert(string str, TokenValue tokVal);

  void insertAllKeywords();

  Terminator terminator;
  map<char, std::unique_ptr<TrieNode>> children;
};

void TrieNode::insertAllKeywords() {
  insert("def", TokenValue::TK_DEF);
  insert("where", TokenValue::TK_WHERE);
  insert("is", TokenValue::TK_IS);
  insert("*", TokenValue::MUL);
  insert("=", TokenValue::EQ);
  insert("(", TokenValue::LP);
  insert(",", TokenValue::COMMA);
  insert(")", TokenValue::RP);
  insert("}", TokenValue::CRP);
  insert("{", TokenValue::CLP);
  insert("+", TokenValue::ADD);
}

void TrieNode::insertImpl(string str, int idx, int size, TokenValue tokVal) {
  if (idx == size)
    return;

  if (children.find(str[idx]) == children.end()) {
    if (idx == size - 1)
      children[str[idx]] = unique_ptr<TrieNode>(new TrieNode(tokVal));
    else
      children[str[idx]] = unique_ptr<TrieNode>(new TrieNode());
  }

  else {
    if (idx == size - 1) {
      children[str[idx]]->terminator.isEnd = true;
      children[str[idx]]->terminator.tokVal = tokVal;
    }
  }
  children[str[idx]]->insertImpl(str, idx + 1, size, tokVal);
}

void TrieNode::insert(string str, TokenValue tokVal) {
  return insertImpl(str, 0, str.size(), tokVal);
}

/// Represent a token.
struct Token {
public:
  Token() = default;
  Token(int64_t num) : tokenAsNumber(num), val(TokenValue::NUM){};
  Token(string str, TokenValue val) : tokenAsString(str), val(val){};
  static void printTokValue(Token val);

  string tokenAsString;
  int64_t tokenAsNumber;
  TokenValue val;
};

class Lexer {
public:
  Lexer() = delete;
  Lexer(string str);

  bool expect(TokenValue val);
  bool nextIs(TokenValue val);
  void putBackCurrTok();
  TokenValue getNextToken();
  TokenValue getCurrToken() { return currTok.val; };
  Token getCurrTok() { return currTok; };
  int getTokPrecedence();

private:
  // parsed string.
  string str;
  // current pos in the string.
  size_t idxPos = 0;
  // current parsed token.
  Token currTok;
  // trie node to store keywords.
  unique_ptr<TrieNode> trieHead;
  // operator precedence.
  llvm::StringMap<int> precedence;
};

Lexer::Lexer(string str) : str(str), trieHead(new TrieNode()) {
  trieHead->insertAllKeywords();
  precedence["*"] = 2;
  precedence["+"] = 1;
}

int Lexer::getTokPrecedence() {
  string tokAsString = currTok.tokenAsString;
  if (precedence.count(tokAsString) == 0)
    return -1;
  return precedence[tokAsString];
}

// XXX: here we do not update 'currTok'
// and it remains to the previous parsed
// value. Use stack?
void Lexer::putBackCurrTok() {
  if (currTok.val == TokenValue::NUM) {
    int nSize = std::to_string(currTok.tokenAsNumber).size();
    idxPos -= nSize;
  } else {
    idxPos -= currTok.tokenAsString.size();
  }
}

TokenValue Lexer::getNextToken() {
  if (idxPos < str.size()) {

    // white spaces.
    while (idxPos < str.size() && isspace(str[idxPos]))
      idxPos++;

    // end of the string.
    if (idxPos == str.size())
      return TokenValue::EOS;

    int number = 0;
    bool isNumber = false;
    while (isdigit(str[idxPos])) {
      number = number * 10 + (str[idxPos++] - '0');
      isNumber = true;
    }
    if (isNumber) {
      currTok = Token(number);
      return TokenValue::NUM;
    }

    TrieNode *currHead = trieHead.get();
    string keyword;
    string identifier;
    for (size_t i = 0;
         i + idxPos < str.size() && (isalpha(str[i + idxPos]) || currHead);
         i++) {
      if (currHead) {
        auto it = currHead->children.find(str[i + idxPos]);
        currHead =
            (it == currHead->children.end()) ? nullptr : it->second.get();
        if (currHead)
          keyword += str[i + idxPos];
        if (currHead && currHead->terminator.isEnd)
          break;
      }
      identifier += str[i + idxPos];
    } // for.
    if (keyword.size()) {
      idxPos += keyword.size();
      currTok = Token(keyword, currHead->terminator.tokVal);
      return currHead->terminator.tokVal;
    }
    if (identifier.size()) {
      idxPos += identifier.size();
      currTok = Token(identifier, TokenValue::NAME);
      return TokenValue::NAME;
    }
  }
  return TokenValue::EOS;
}

bool Lexer::expect(TokenValue tokVal) {
  auto currTok = getNextToken();
  return currTok == tokVal;
}

bool Lexer::nextIs(TokenValue tokVal) {
  auto currTokVal = getNextToken();
  return currTokVal == tokVal;
}

void Token::printTokValue(Token tok) {
  if (tok.val == TokenValue::TK_DEF)
    cout << "TK_DEF\n";
  if (tok.val == TokenValue::TK_IS)
    cout << "TK_IS\n";
  if (tok.val == TokenValue::TK_WHERE)
    cout << "TK_WHERE\n";
  if (tok.val == TokenValue::NAME)
    cout << "NAME: " << tok.tokenAsString << "\n";
  if (tok.val == TokenValue::LP)
    cout << ")\n";
  if (tok.val == TokenValue::RP)
    cout << ")\n";
  if (tok.val == TokenValue::CLP)
    cout << "{\n";
  if (tok.val == TokenValue::CRP)
    cout << "}\n";
  if (tok.val == TokenValue::EQ)
    cout << "=\n";
  if (tok.val == TokenValue::EOS)
    cout << "EOS\n";
}

} // end namespace lexer.

namespace parser {

using namespace lexer;

/// End result of the parsing.
class Expression {
public:
  Expression() = delete;
  Expression(string funcName, Expr *rhs, Token assignment, Expr *lhs,
             ScopedContext &ctx)
      : funcName(funcName), rhs(rhs), assignment(assignment), lhs(lhs),
        ctx(ctx){};

  Expr *getLhs() { return lhs; };
  Expr *getRhs() { return rhs; };
  string getFuncName() { return funcName; };

  void print();

private:
  string funcName;
  Expr *rhs;
  Token assignment;
  Expr *lhs;
  // vector<const Operand *> operands;
  ScopedContext &ctx;
};

void Expression::print() {
  cout << "---------------------------------\n";
  cout << "def with name: " << funcName << "\n";
  lhs->walk();
  cout << "\n";
  cout << "  ";
  Token::printTokValue(assignment);
  rhs->walk(4);
  cout << "\n";
  cout << "---------------------------------\n";
}

class Parser {
public:
  Parser() = delete;
  Parser(string str, ScopedContext &ctx) : lex(Lexer(str)), ctx(ctx){};

  Expression parseFunction();
  ScopedContext &getCtx() { return ctx; };

private:
  Lexer lex;
  ScopedContext &ctx;

  // flag to indicate if we need to fold operations within other operations of
  // the same type. The flag is set to false if we hit a LP.
  bool foldExpr = true;

  bool parseFuncName(string &str);
  bool parseOperands(vector<ParsedOperand> &operands);
  bool parseOperand(ParsedOperand &operand);
  void parseWhereClause(vector<ParsedOperand> &operands);
  Expression parseStmt(vector<ParsedOperand> &operands, string funcName);
  Expr *parsePrimary(vector<ParsedOperand> &operands);
  Expr *parseExpression(vector<ParsedOperand> &operands);
  Expr *parseRhs(int precedence, Expr *lhs, vector<ParsedOperand> &operands);
  Expr *buildOperand(string id, vector<ParsedOperand> &operands);
};

bool Parser::parseFuncName(string &str) {
  auto currTok = lex.getNextToken();
  str = lex.getCurrTok().tokenAsString;
  return currTok == TokenValue::NAME;
}

bool Parser::parseOperand(ParsedOperand &operand) {
  // parse type.
  if (!lex.expect(TokenValue::NAME))
    return false;
  operand.type = lex.getCurrTok().tokenAsString;

  // parse dimensions (32, 32). Only 2d atm.
  SmallVector<int64_t, 2> dims;
  if (!lex.expect(TokenValue::LP))
    return false;
  if (!lex.expect(TokenValue::NUM))
    return false;
  dims.push_back(lex.getCurrTok().tokenAsNumber);
  if (!lex.expect(TokenValue::COMMA))
    return false;
  if (!lex.expect(TokenValue::NUM))
    return false;
  dims.push_back(lex.getCurrTok().tokenAsNumber);
  if (!lex.expect(TokenValue::RP))
    return false;
  assert(dims.size() == 2 && "expect two dimensions only");
  operand.dims = dims;

  // parse identifier.
  if (!lex.expect(TokenValue::NAME))
    return false;
  operand.name = lex.getCurrTok().tokenAsString;
  return true;
}

bool Parser::parseOperands(vector<ParsedOperand> &operands) {
  auto currTok = lex.getNextToken();
  if (currTok != TokenValue::LP)
    return false;
  while (currTok != TokenValue::RP) {
    ParsedOperand operand;
    if (!parseOperand(operand))
      return false;
    operands.push_back(operand);
    currTok = lex.getNextToken();
    if (currTok == TokenValue::RP)
      break;
    assert(currTok == TokenValue::COMMA);
  }
  return true;
}

// TODO: We always build a matrix. Today works as we have just matrices
// tomorrow we may want to extend this.
Expr *Parser::buildOperand(string id, vector<ParsedOperand> &operands) {
  Expr *operandExpr = nullptr;
  for (auto operand : operands)
    if (operand.name == id)
      operandExpr = new Matrix(id, operand.dims);
  assert(operandExpr && "cannot find operand!");
  return operandExpr;
}

Expr *Parser::parsePrimary(vector<ParsedOperand> &operands) {
  auto currTok = lex.getNextToken();
  switch (currTok) {
  case TokenValue::NAME:
    return buildOperand(lex.getCurrTok().tokenAsString, operands);
  case TokenValue::LP: {
    // do not fold. The user inserted parenthesis. Respect them.
    foldExpr = false;
    Expr *term = parseExpression(operands);
    currTok = lex.getNextToken();
    if (currTok != TokenValue::RP)
      assert(0 && "expect RP");
    return term;
  }
  default:
    assert(0 && "only name supported");
  }
}

Expr *Parser::parseRhs(int precedence, Expr *lhs,
                       vector<ParsedOperand> &operands) {
  auto currTok = lex.getCurrToken();
  while (true) {
    switch (currTok) {
    case TokenValue::MUL:
      lhs = mul(foldExpr, lhs, parsePrimary(operands));
      currTok = lex.getNextToken();
      break;
    case TokenValue::ADD: {
      int currPrec = lex.getTokPrecedence();
      Expr *rhs = parsePrimary(operands);
      (void)lex.getNextToken();
      int nextPrec = lex.getTokPrecedence();
      if (currPrec < nextPrec) {
        rhs = parseRhs(currPrec + 1, rhs, operands);
      } else { // put back if the precedence is same.
        lex.putBackCurrTok();
      }
      lhs = add(foldExpr, lhs, rhs);
      currTok = lex.getNextToken();
      break;
    }
    default:
      // before returing put back
      // last token.
      lex.putBackCurrTok();
      return lhs;
    }
  }
}

Expr *Parser::parseExpression(vector<ParsedOperand> &operands) {
  Expr *lhs = parsePrimary(operands);
  (void)lex.getNextToken();
  return parseRhs(0, lhs, operands);
}

void Parser::parseWhereClause(vector<ParsedOperand> &operands) {
  do {
    if (!lex.expect(TokenValue::NAME))
      assert(0 && "expect an identifier");
    string idArray = lex.getCurrTok().tokenAsString;
    if (!lex.expect(TokenValue::TK_IS))
      assert(0 && "expect TK_IS");

    vector<string> properties;
    do {
      if (!lex.expect(TokenValue::NAME))
        assert(0 && "expect a name");
      properties.push_back(lex.getCurrTok().tokenAsString);
    } while (lex.nextIs(TokenValue::COMMA));

    // update the parsed operands.
    bool found = false;
    for (auto &operand : operands) {
      if (operand.name == idArray) {
        found = true;
        for (auto property : properties)
          operand.properties.push_back(
              ParsedOperand::convertProperty(property));
      }
    }
    assert(found && "id not found");
  } while (lex.nextIs(TokenValue::COMMA));
}

static void collectOperandsImpl(Expr *root, map<string, Operand *> &map) {
  if (auto *binaryExpr = llvm::dyn_cast_or_null<NaryExpr>(root)) {
    for (auto *child : binaryExpr->getChildren())
      collectOperandsImpl(child, map);
  }
  if (auto *unaryExpr = llvm::dyn_cast_or_null<UnaryExpr>(root)) {
    collectOperandsImpl(unaryExpr->getChild(), map);
  }
  if (Operand *operand = llvm::dyn_cast_or_null<Operand>(root)) {
    map[operand->getName()] = operand;
  }
}

static map<string, Operand *> collectOperands(Expr *lhs, Expr *rhs) {
  map<string, Operand *> operands;
  if (Operand *lhsOperand = llvm::dyn_cast_or_null<Operand>(lhs))
    operands[lhsOperand->getName()] = lhsOperand;
  collectOperandsImpl(rhs, operands);
  return operands;
}

Expression Parser::parseStmt(vector<ParsedOperand> &operands, string funcName) {
  if (!lex.expect(TokenValue::NAME))
    assert(0 && "expect lhs name");
  Expr *lhs = buildOperand(lex.getCurrTok().tokenAsString, operands);
  if (!lex.expect(TokenValue::EQ))
    assert(0 && "expect assingment EQ");
  Token assignment = lex.getCurrTok();
  Expr *rhs = parseExpression(operands);

  if (lex.nextIs(TokenValue::TK_WHERE))
    parseWhereClause(operands);

  // update built expression with new parsed operand properties.
  map<string, Operand *> operandsExpr = collectOperands(lhs, rhs);
  assert(operandsExpr.size() == operands.size());
  for (auto operand : operands)
    operandsExpr[operand.name]->setProperties(operand.properties);

  return Expression(funcName, rhs, assignment, lhs, getCtx());
}

// TODO: better error message. Asserting does not say much.
Expression Parser::parseFunction() {
  if (!lex.expect(TokenValue::TK_DEF))
    assert(0 && "expect TK_DEF");
  string funcName;
  if (!parseFuncName(funcName))
    assert(0 && "expect function name");
  vector<ParsedOperand> operands;
  if (!parseOperands(operands))
    assert(0 && "failed to parse operands");
  assert(operands.size() > 0 && "expect one or more operands");
  if (!lex.expect(TokenValue::CLP))
    assert(0 && "expect CLP");
  return parseStmt(operands, funcName);
}
} // end namespace parser.

namespace {
bool isSameTree(const Expr *root1, const Expr *root2) {
  if (!root1 && !root2)
    return true;
  else if (root1 && root2) {
    if (root1->getKind() != root2->getKind()) {
      return false;
    }
    if (llvm::isa<Operand>(root1)) {
      const auto *root1Operand = llvm::dyn_cast_or_null<Operand>(root1);
      const auto *root2Operand = llvm::dyn_cast_or_null<Operand>(root2);
      if (root1Operand->getName() != root2Operand->getName()) {
        return false;
      }
      if (root1Operand->getShape() != root2Operand->getShape()) {
        return false;
      }
      return true;
    }
    // unary.
    if (llvm::isa<UnaryExpr>(root1) && llvm::isa<UnaryExpr>(root2)) {
      const UnaryExpr *tree1Op = llvm::dyn_cast_or_null<UnaryExpr>(root1);
      const UnaryExpr *tree2Op = llvm::dyn_cast_or_null<UnaryExpr>(root2);
      // different unaries op.
      if (tree1Op->getKind() != tree2Op->getKind())
        return false;
      return isSameTree(tree1Op->getChild(), tree2Op->getChild());
    }
    // binary.
    if (llvm::isa<NaryExpr>(root1) && llvm::isa<NaryExpr>(root2)) {
      const NaryExpr *tree1Op = llvm::dyn_cast_or_null<NaryExpr>(root1);
      const NaryExpr *tree2Op = llvm::dyn_cast_or_null<NaryExpr>(root2);
      // different binary ops.
      if (tree1Op->getKind() != tree2Op->getKind())
        return false;
      // different number of children.
      if (tree1Op->getChildren().size() != tree2Op->getChildren().size())
        return false;
      int numberOfChildren = tree1Op->getChildren().size();
      for (int i = 0; i < numberOfChildren; i++)
        if (!isSameTree(tree1Op->getChildren()[i], tree2Op->getChildren()[i]))
          return false;
      return true;
    }
  }
  return false;
}
} // namespace

TEST(Parser, simpleMul) {
  using namespace parser;
  ScopedContext ctx;
  string s = R"(
  def mul(float(32, 32) A, float(32, 32) B, float(32, 32) C) {
    C = A * B
  })";

  Parser p(s, ctx);
  auto root = p.parseFunction();
  assert(root.getRhs() && "must be non-null");
  auto *a = new Matrix("A", {32, 32});
  auto *b = new Matrix("B", {32, 32});
  auto *truth = mul(a, b);
  EXPECT_EQ(isSameTree(root.getRhs(), truth), true);
}

TEST(Parser, simpleAdd) {
  using namespace parser;
  ScopedContext ctx;
  string s = R"(
  def add(float(32, 32) A, float(32, 32) B, float(32, 32) C) {
    C = A + B
  })";

  Parser p(s, ctx);
  auto root = p.parseFunction();
  assert(root.getRhs() && "must be non-null");
  auto *a = new Matrix("A", {32, 32});
  auto *b = new Matrix("B", {32, 32});
  auto *truth = add(a, b);
  EXPECT_EQ(isSameTree(root.getRhs(), truth), true);
}

TEST(Parser, simplePrecedence) {
  using namespace parser;
  ScopedContext ctx;
  string s = R"(
  def precedence(float(32, 32) A, float(32, 32) B, float(32, 32) C) {
    C = A + B * A
  })";

  Parser p(s, ctx);
  auto root = p.parseFunction();
  root.print();
  assert(root.getRhs() && "must be non-null");
  auto *a = new Matrix("A", {32, 32});
  auto *b = new Matrix("B", {32, 32});
  auto *truth = add(a, mul(b, a));
  EXPECT_EQ(isSameTree(root.getRhs(), truth), true);
}

TEST(Parser, variadicMul) {
  using namespace parser;
  ScopedContext ctx;
  string s = R"(
  def mul(float(32, 32) A, float(32, 32) B, 
          float(32, 32) C, float(32, 32) D,
          float(32, 32) E, float(32, 32) F, 
          float(32, 32) G) {
    G = A * B * C * D * E * F
  })";

  Parser p(s, ctx);
  auto root = p.parseFunction();
  assert(root.getRhs() && "must be non null");
  auto *a = new Matrix("A", {32, 32});
  auto *b = new Matrix("B", {32, 32});
  auto *c = new Matrix("C", {32, 32});
  auto *d = new Matrix("D", {32, 32});
  auto *e = new Matrix("E", {32, 32});
  auto *f = new Matrix("F", {32, 32});
  auto *truth = mul(a, b, c, d, e, f);
  EXPECT_EQ(isSameTree(root.getRhs(), truth), true);
}

TEST(Parser, variadicAdd) {
  using namespace parser;
  ScopedContext ctx;
  string s = R"(
  def mul(float(32, 32) A, float(32, 32) B, 
          float(32, 32) C, float(32, 32) D,
          float(32, 32) E, float(32, 32) F, 
          float(32, 32) G) {
    G = A + B + C + D + E + F
  })";

  Parser p(s, ctx);
  auto root = p.parseFunction();
  assert(root.getRhs() && "must be non null");
  auto *a = new Matrix("A", {32, 32});
  auto *b = new Matrix("B", {32, 32});
  auto *c = new Matrix("C", {32, 32});
  auto *d = new Matrix("D", {32, 32});
  auto *e = new Matrix("E", {32, 32});
  auto *f = new Matrix("F", {32, 32});
  auto *truth = add(a, b, c, d, e, f);
  EXPECT_EQ(isSameTree(root.getRhs(), truth), true);
}

TEST(Parser, whereClause) {
  // TODO: There is a bug here. ConvertProperty hits the unrechable.
  /*
    using namespace parser;
    ScopedContext ctx;
    string s = R"(
    def mul(float(32, 32) A, float(32, 32) B, float(32, 32) C) {
      C = A * B
      where A is SYMMETRIC, SQUARE, B is SYMMETRIC, C is SYMMETRIC
    })";

    Parser p(s, ctx);
    auto root = p.parseFunction();
    assert(root.getRhs() && "must be non-null");
    auto *A = new Matrix("A", {32, 32});
    A->setProperties({Expr::ExprProperty::SYMMETRIC,
    Expr::ExprProperty::SQUARE}); auto *B = new Matrix("B", {32, 32});
    B->setProperties({Expr::ExprProperty::SQUARE});
    auto *truth = mul(A, B);
    EXPECT_EQ(isSameTree(root.getRhs(), truth), true);
  */
}

TEST(Parser, paren) {
  using namespace parser;
  ScopedContext ctx;
  string s = R"(
  def mul(float(32, 32) A, float(32, 32) B, float(32, 32) C) {
    C = A * (A * B) * ((A * B))
  })";

  Parser p(s, ctx);
  auto root = p.parseFunction();
  assert(root.getRhs() && "must be non-null");
  auto *a = new Matrix("A", {32, 32});
  auto *b = new Matrix("B", {32, 32});
  // must be non equal as we keep parenthesis into account.
  auto *truth = mul(a, a, b, a, b);
  EXPECT_EQ(!isSameTree(root.getRhs(), truth), true);
}

TEST(Parser, parenAdd) {
  using namespace parser;
  ScopedContext ctx;
  string s = R"(
  def mul(float(32, 32) A, float(32, 32) B, float(32, 32) C) {
    C = C + ((A + A) + (A + B)) + (((A + C)))
  })";

  Parser p(s, ctx);
  auto root = p.parseFunction();
  assert(root.getRhs() && "must be non-null");
  auto *a = new Matrix("A", {32, 32});
  auto *b = new Matrix("B", {32, 32});
  // must be non equal as we keep parenthesis into account.
  auto *truth = mul(a, a, b, a, b);
  EXPECT_EQ(!isSameTree(root.getRhs(), truth), true);
}
