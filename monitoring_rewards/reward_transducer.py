from typing import Sequence

import graphviz
from flloat.parser.ltlf import LTLfParser
from pythomata.core import SymbolType
from pythomata.impl.symbolic import SymbolicDFA
from sympy import Symbol
from sympy.logic.boolalg import And, Or, Not
from sympy.logic.boolalg import BooleanAtom

from monitoring_rewards.core import Reward, TraceStep


class RewardTransducer(SymbolicDFA):
    """
    A Mealy reward transducer
    """

    def __init__(self, ltlf_formula: str, r, c, s, f):
        """
        :param ltlf_formula: A flloat-compatible LTLf formula
        :param r: (reward) given to the agent when the formula is temporarily true in the current
        partial trace
        :param c: (cost) given to the agent when the formula is temporarily false in the current
        partial trace
        :param s: (success) value given to the agent when the formula is permanently true in the current
        partial trace
        :param f: (failure) given to the agent when the formula is permanently false in the current
        partial trace
        """
        SymbolicDFA.__init__(self)
        self._init_symbolic_dfa(ltlf_formula)

        self.r = r
        self.c = c
        self.s = s
        self.f = f

        self.output_function = self._build_output_function()
        self.trace: Sequence[SymbolType] = []

    def _init_symbolic_dfa(self, ltlf_formula: str):
        """
        Initialize the SymbolicDFA
        :param ltlf_formula: the LTLf formula
        """
        parser = LTLfParser()
        self.parsed_formula = parser(ltlf_formula)
        symbolic_dfa = self.parsed_formula.to_automaton()

        for state in symbolic_dfa.states:
            self.states.add(state)

        self.set_initial_state(symbolic_dfa.initial_state)

        for state in symbolic_dfa.accepting_states:
            self.set_accepting_state(state, is_accepting=True)

        for transition in symbolic_dfa.get_transitions():
            self.add_transition(transition)

    def __call__(self, trace_step: TraceStep) -> (Reward, bool):
        """
        Takes a trace step and gives a reward
        :param trace_step: a trace step
        :return: the reward associated to taking the given trace step considering the whole current trace
        """
        self.trace.append(trace_step)

        return self.get_reward(trace=self.trace), self.is_trace_perm()

    def is_state_perm(self, state) -> bool:
        """
        Tells if the given state is inescapable
        :param state: the state to evaluate
        :return: if the state is inescapable or not
        """
        transitions_from = self.get_transitions_from(state)
        if len(transitions_from) == 1:
            from_state, guard, to_state = transitions_from.pop()
            if isinstance(guard, BooleanAtom):
                return True

        return False

    def is_trace_perm(self) -> bool:
        """
        Tells if the current trace brought the underlying automata to an inescapable state
        :return: if the state to which the underlying automata goes after traversing the whole current trace is
        inescapable or not
        """
        state = self.initial_state
        for symbol in self.trace:
            state = self.get_successor(state, symbol)

            if self.is_state_perm(state):
                return True

        return False

    def _clear_trace(self):
        """
        Empty the trace.
        """
        self.trace = []

    def reset(self):
        """
        Resets the transducer to its initial state.
        """
        self._clear_trace()

    def get_reward(self, trace) -> Reward:
        """
        Compute the reward given a trace
        :param trace: the trace
        :return: the reward
        """
        current_state = self.initial_state

        for symbol in trace:
            current_state = self.get_successor(current_state, symbol)

            if self.is_state_perm(current_state):
                if self.accepts(trace):
                    return self.s
                else:
                    return self.f

        if self.parsed_formula.truth(trace, 0):
            return self.r
        else:
            return self.c

    def _build_output_function(self):
        trace = []
        output_function = {}

        def explore_node(state):
            transitions = self.get_transitions_from(state)
            for transition in transitions:
                from_state, guard, to_state = transition

                if (from_state, guard) in output_function.keys():
                    return

                if isinstance(guard, BooleanAtom):
                    trace.append({})
                if isinstance(guard, (And, Or)):
                    trace_step = {}
                    for symbol in guard.args:
                        if isinstance(symbol, Symbol):
                            trace_step[str(symbol)] = True
                        if isinstance(symbol, Not):
                            trace_step[str(symbol.args[0])] = False
                    trace.append(trace_step)

                if isinstance(guard, Symbol):
                    trace.append({str(guard): True})

                if isinstance(guard, Not):
                    trace.append({str(guard.args[0]): False})

                if self.is_state_perm(to_state):
                    if self.parsed_formula.truth(trace, 0):
                        output_function[from_state, guard] = self.s
                    else:
                        output_function[from_state, guard] = self.f
                else:
                    if self.parsed_formula.truth(trace, 0):
                        output_function[from_state, guard] = self.r
                    else:
                        output_function[from_state, guard] = self.c

                if from_state != to_state:
                    explore_node(to_state)
                else:
                    trace.pop()

            if len(trace) > 0:
                trace.pop()

        explore_node(self.initial_state)

        return output_function

    def to_graphviz(self) -> graphviz.Digraph:
        graph = graphviz.Digraph(format="svg")
        graph.node("fake", style="invisible")
        for state in self.states:
            if state == self.initial_state:
                if state in self.accepting_states:
                    graph.node(str(state), root="true", shape="doublecircle")
                else:
                    graph.node(str(state), root="true")
            elif state in self.accepting_states:
                graph.node(str(state), shape="doublecircle")
            else:
                graph.node(str(state))

        graph.edge("fake", str(self.initial_state), style="bold")

        for (start, guard, end) in self.get_transitions():
            guard = str(guard) + ' / ' + str(self.output_function[start, guard])
            graph.edge(str(start), str(end), label=str(guard))

        return graph
