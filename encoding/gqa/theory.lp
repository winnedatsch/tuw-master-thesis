
% idea: scene objects encoded like:
% object(ID)
% has_attribute(ID, class, car)
% has_attribute(ID, color, red)
% has_attribute(ID, size, large)
% has_attribute(ID, any, full)
% has_relation(ID, ID', in)
% has_relation(ID, ID'', in_front_of)

% ------- Rules for operations -------
% scene
% state(0,ID) :- object(ID).
state(T,ID) :- scene(T), object(ID).

% all_different
% all_same

% and
bool(T,yes) :- and(T, RT0, RT1), bool(RT0,yes), bool(RT1,yes).
bool(T,no) :- and(T, RT0, RT1), not bool(T,yes).

% choose_attr
attr(T, VALUE) :- choose_attr(T, RT, ATTR, VALUE, VALUE'), state(RT, ID), has_attribute(ID, ATTR, VALUE).
attr(T, VALUE') :- choose_attr(T, RT, ATTR, VALUE, VALUE'), state(RT, ID), has_attribute(ID, ATTR, VALUE').

% choose_rel
rel(T, REL) :- choose_rel(T, RT, CLASS, REL, REL'), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID, REL, ID').
rel(T, REL') :- choose_rel(T, RT, CLASS, REL, REL'), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID, REL', ID').

% compare

% exist
bool(T,yes) :- exist(T, RT), state(RT,ID).
bool(T,no) :- exist(T, RT), not bool(T,yes).

% filter
state(T,ID) :- filter(T, RT, ATTR, VALUE), state(RT, ID), has_attribute(ID, ATTR, VALUE).

% or
bool(T,yes) :- or(T, RT0, RT1), bool(RT0,yes).
bool(T,yes) :- or(T, RT0, RT1), bool(RT1,yes).
bool(T,no) :- or(T, RT0, RT1), not bool(T,yes).

% query_attr
attr(T,VALUE) :- query_attr(T, RT, ATTR), state(RT, ID), has_attribute(ID, ATTR, VALUE).

% relate
state(T, ID') :- relate(T, RT, CLASS, REL), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID', REL, ID).

% relate_any
state(T, ID') :- relate_any(T, RT, REL), state(RT, ID), has_relation(ID', REL, ID).

% select
state(T,ID) :- select(T, RT, CLASS), state(RT, ID), has_attribute(ID, class, CLASS).

% two_different
bool(T, yes) :- two_different(T, RT0, RT1, ATTR), state(RT0, ID), state(RT1, ID'), has_attribute(ID, ATTR, VALUE), has_attribute(ID', ATTR, VALUE'), VALUE != VALUE'.
bool(T, yes) :- two_different(T, RT0, RT1, ATTR), state(RT0, ID), state(RT1, ID'), has_attribute(ID, ATTR, _), not has_attribute(ID', ATTR, _).
bool(T, yes) :- two_different(T, RT0, RT1, ATTR), state(RT0, ID), state(RT1, ID'), not has_attribute(ID, ATTR, _), has_attribute(ID', ATTR, _).
bool(T,no) :- two_different(T, RT0, RT1, ATTR), not bool(T,yes).

% two_same
bool(T, yes) :- two_same(T, RT0, RT1, ATTR), state(RT0, ID), state(RT1, ID'), has_attribute(ID, ATTR, VALUE), has_attribute(ID', ATTR, VALUE'), VALUE == VALUE'.
bool(T,no) :- two_same(T, RT0, RT1, ATTR), not bool(T,yes).

% unique
state(T,ID) :- unique(T, RT), state(RT,ID).
:- unique(T, RT), state(RT,ID), state(RT,ID'), ID!=ID'.

% verify_attr
bool(T, yes) :- verify_attr(T, RT, ATTR, VALUE), state(RT, ID), has_attribute(ID, ATTR, VALUE).
bool(T,no) :- verify_attr(T, RT, ATTR, VALUE), not bool(T,yes).

% verify_rel
bool(T, yes) :- verify_rel(T, RT, CLASS, REL), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID, ID', REL).
bool(T,no) :- verify_rel(T, RT, CLASS, REL), not bool(T,yes).


% Derive answer (T must equal the last point in time)
ans(V) :- end(T), attr(T,V).
ans(V) :- end(T), rel(T,V).
ans(V) :- end(T), bool(T,V).

% :- not ans(_).

#show ans/1.
#show state/2.